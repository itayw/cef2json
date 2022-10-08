#[macro_use]
extern crate lazy_static;

use crossbeam_channel::bounded;
use crossbeam_queue::ArrayQueue;
use num_format::{Locale, ToFormattedString};
use ocl::builders::ContextProperties;
use ocl::prm::cl_uchar;
use ocl::{core, flags, Platform};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::hash::Hash;
use std::time::{Duration, Instant};

const STAT_INTERVAL: u64 = 3;

fn main() {
    let mut devices = vec![];
    let mut platform_id = Platform::default();
    let platforms = Platform::list();
    for platform in platforms {
        match core::get_device_ids(&platform, Some(ocl::flags::DEVICE_TYPE_GPU), None) {
            Ok(device_ids) => {
                devices = [devices, device_ids].concat();
                platform_id = platform;
            }
            Err(_) => {}
        };
    }

    println!(
        "Found {} GPU devices, will use only 1 for this POC, compiling kernel...",
        devices.len()
    );
    let src = r#"
                __kernel void kernel_cefparser(__global uchar * in_out) {
                    ulong idx = get_global_id(0);
                
                    uchar prev_char_to_inspect = in_out[idx-1];
                    uchar char_to_inspect = in_out[idx];

                    uchar is_ln = !(char_to_inspect - '\n');
                    uchar is_space = !(char_to_inspect - ' ') << 1;
                    uchar is_pipe = !(char_to_inspect - '|') << 2;
                    uchar is_eq = (!(char_to_inspect - '=') && (prev_char_to_inspect - '\\')) << 3 ;

                    if (is_eq) {
                        uint distance = 0;
                        bool is_part_start = false;
                        for (int i = idx ; i > 0 ; i--) {
                            if (in_out[i] == ' ' || in_out[i] == '|') {
                                is_part_start = true;
                                break;
                            }
                            else if (in_out[i] == 4 || in_out[i] == 2) {
                                is_part_start = true;
                                break;
                            }
                            distance++;
                        }
                        
                        if (is_part_start) {
                            in_out[idx-distance] = 16;
                        }
                    }

                    uchar result = is_space | is_ln | is_pipe | is_eq;
                    if (result == 0) {
                        result = char_to_inspect;
                    }
                    in_out[idx] = result;
                }
            "#;
    let src_cstring = CString::new(src).unwrap();

    //let q = ArrayQueue::new(1_000_000);
    let s: crossbeam_channel::Sender<(Vec<u8>, Vec<&str>)>;
    let r: crossbeam_channel::Receiver<(Vec<u8>, Vec<&str>)>;
    (s, r) = bounded(10_000_000);

    std::thread::spawn(move || {
        for _ in 1..4 {
            loop {
                let (in_out, work_stdin) = r.recv().unwrap();
                let mut msg_idx = 0;
                for msg in in_out.split(|x| *x == 1) {
                    if msg.len() == 0 {
                        continue;
                    }

                    //println!("test");

                    let pos: Vec<usize> = msg
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, val)| if *val == 8 { Some(idx) } else { None })
                        .collect();

                    let mut json = "{".to_string();
                    for (i, p) in pos.iter().enumerate() {
                        let mut start = 0;
                        let mut end = 0;

                        for i in (0..*p).rev() {
                            if msg[i] == 4 || msg[i] == 2 {
                                start = i + 1;
                                break;
                            }
                        }

                        if i + 1 >= pos.len() {
                            end = work_stdin[msg_idx].len();
                        } else {
                            let last = pos[i + 1];
                            for i in (*p..last).rev() {
                                if msg[i] == 2 {
                                    end = i;
                                    break;
                                }
                            }
                            if end == 0 {
                                end = work_stdin[msg_idx].len();
                            }
                        }

                        json.push_str("\"");
                        json.push_str(&work_stdin[msg_idx][start..*p]);
                        json.push_str("\":");
                        json.push_str("\"");
                        json.push_str(&work_stdin[msg_idx][*p + 1..end]);
                        json.push_str("\",");
                    }
                    json.truncate(json.len() - 1);
                    json.push('}');
                    //println!("{}", json);
                    // });

                    msg_idx += 1;
                }

                //println!("{}", r.recv().unwrap());
            }
            //std::thread::sleep(Duration::from_millis(1));
        }
    });

    devices
        .into_par_iter()
        //.skip(1)
        .take(1)
        .for_each(move |device_id| {
            let mut start = Instant::now();

            let mut total_counter = 0;
            let mut total_bytes: f32 = 0.0;

            println!("device {:?} loading", device_id);
            let context_properties = ContextProperties::new().platform(platform_id);
            let device_spec = ocl::builders::DeviceSpecifier::from(device_id);
            let device = ocl::Device::from(device_id);
            let context = ocl::Context::builder()
                .properties(context_properties)
                .devices(device_spec)
                .build()
                .unwrap();

            let queue = ocl::Queue::new(&context, device, None).unwrap();
            let program = ocl::Program::with_source(
                &context,
                &[src_cstring.clone()],
                Some(&[device]),
                &CString::new("").unwrap(),
            )
            .unwrap();

            let kernel = ocl::Kernel::builder()
                .program(&program)
                .name("kernel_cefparser")
                .arg_named("in_out", None::<&ocl::Buffer<cl_uchar>>)
                .build()
                .unwrap();

            let chunk_size = 1;
            let mut counter = 0;
            let mut bytes = 0;

            loop {
                let work_stdin = get_work(chunk_size as u32);
                let _chunk_size = work_stdin.len();

                if _chunk_size > 0 {
                    let mut in_out = flatten_strings(work_stdin.clone().into_iter());
                    let in_out_buf = ocl::Buffer::builder()
                        .context(&context)
                        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
                        .copy_host_slice(&in_out)
                        .len(in_out.len())
                        .build()
                        .unwrap();
                    let work_size = in_out.len();

                    kernel.set_arg("in_out", &in_out_buf).unwrap();

                    unsafe {
                        kernel
                            .cmd()
                            .global_work_size(
                                ocl::SpatialDims::new(Some(work_size as usize), None, None)
                                    .unwrap(),
                            )
                            /*.local_work_size(
                                ocl::SpatialDims::new(Some(32 as usize), Some(1 as usize), None)
                                    .unwrap(),
                            )*/
                            .queue(&queue)
                            .enq()
                            .unwrap();
                        in_out_buf.read(&mut in_out).queue(&queue).enq().unwrap();
                        println!("in {:?}", in_out);
                    }

                    in_out.par_split(|x| *x == 1).for_each(|msg| {
                        if msg.len() == 0 {
                            return;
                        }
                        for keypair in msg.split(|x| *x == 16) {
                            println!(
                                "keypair {}",
                                keypair.iter().map(|x| *x as char).collect::<String>()
                            );
                        }
                    });

                    //println!("{:?}", in_out);

                    /*
                        let mut msg_idx = 0;

                        for msg in in_out.split(|x| *x == 1) {
                            if msg.len() == 0 {
                                continue;
                            }

                            let pos: Vec<usize> = msg
                                .iter()
                                .enumerate()
                                .filter_map(|(idx, val)| if *val == 8 { Some(idx) } else { None })
                                .collect();

                            let mut json = "{".to_string();
                            for (i, p) in pos.iter().enumerate() {
                                let mut start = 0;
                                let mut end = 0;

                                for i in (0..*p).rev() {
                                    if msg[i] == 4 || msg[i] == 2 {
                                        start = i + 1;
                                        break;
                                    }
                                }

                                if i + 1 >= pos.len() {
                                    end = work_stdin[msg_idx].len();
                                } else {
                                    let last = pos[i + 1];
                                    for i in (*p..last).rev() {
                                        if msg[i] == 2 {
                                            end = i;
                                            break;
                                        }
                                    }
                                }

                                json.push_str("\"");
                                json.push_str(&work_stdin[msg_idx][start..*p]);
                                json.push_str("\":");
                                json.push_str("\"");
                                json.push_str(&work_stdin[msg_idx][*p + 1..end]);
                                json.push_str("\",");
                            }
                            json.truncate(json.len() - 1);
                            json.push('}');
                            //println!("{}", json);
                            msg_idx += 1;
                        }
                    }
                     */
                    let duration = start.elapsed();
                    total_counter += chunk_size;
                    total_bytes += work_size as f32 / 1024.0 / 1024.0 / 1024.0;

                    counter += chunk_size;
                    bytes += work_size;

                    if duration > Duration::from_secs(STAT_INTERVAL) {
                        eprintln!(
                            "Processed: {} events/{} gb, EPS {} events/second, GBPS {} gb/second",
                            total_counter.to_formatted_string(&Locale::en),
                            total_bytes,
                            counter as f32 / duration.as_secs_f32(),
                            bytes as f32 / duration.as_secs_f32() / 1024.0 / 1024.0 / 1024.0,
                        );
                        start = Instant::now();
                        counter = 0;
                        bytes = 0;
                    }
                }
            }
        });
    //println!("GPU done, elapsed: {:?}ms.", start.elapsed().as_millis());
}

lazy_static! {
    static ref CEF_SAMPLE: Vec<&'static str> = {
        let cef_sample: Vec<&'static str> = [

        "<134>2022-02-14T03:17:30-08:00 TEST CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=איתי act=",
        "<134>Feb 14 19:04:54 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>Feb 14 19:04:54 TEST CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>2022-02-14T03:17:30-08:00 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>2022-02-14T03:17:30-08:00 127.0.0.1 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act= cs1Label=Testing CS1 Label cs1=CS1 Label Value",
        "<134>Feb 14 19:04:54 www.google.com CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>Feb 14 19:04:54 test.hq.ibm.com CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>2022-02-14T03:17:30-08:00 test.hq.example.com CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>2022-02-14T03:17:30-08:00 2001:db8:3333:4444:5555:6666:7777:8888 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>Feb 14 19:04:54 2001:db8:3333:4444:5555:6666:7777:8888 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>2022-02-14T03:17:30-08:00 ::1 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>Feb 14 19:04:54 ::1 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>2022-02-14T03:17:30-08:00 ::1 CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|et= src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act= cs1Label=X cs1=",
        "Feb 14 19:04:54 test.hq.ibm.com CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act= cs1Label=Testing CS1 Label cs1=CS1 Label Value",
        "2022-02-14T03:17:30-08:00 TEST CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act= cs1Label=Testing CS1 Label cs1=CS1 Label Value",
        "<133>Feb 14 2022 05:15:47 135.181.193.106 CEF:0|Trend Micro|Apex Central|2019|WB:7|7|3|deviceExternalId=38 rt=Nov 15 2017 08:43:57 GMT+00:00 app=17 cntLabel=AggregatedCount cnt=1 dpt=80 act=1 src=10.1.128.46 cs1Label=SLF_PolicyName cs1=External User Policy deviceDirection=2 cat=7 dvchost=ApexOneClient08 fname=test.txt request=http://www.violetsoft.net/counter/insert.php?dbserver\\=db1&c_pcode\\=25&c_pid\\=funpop1&c_kind\\=4&c_mac\\=FE-ED-BE-EF-0C-E1 deviceFacility=Apex One shost=ABC-HOST-WKS12",
        "<134>1 Feb 14 19:04:54 test.hq.ibm.com CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
        "<134>1 2022-02-14T03:17:30-08:00 test.hq.example.com CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act="
    ].to_vec();
        let mut buffer: Vec<&'static str> = vec![].into_iter().collect();
        for _ in 0..10_000_000 {
            buffer.push(cef_sample[rand::thread_rng().gen_range(0..cef_sample.len())]);
        }
        buffer
    };
}

fn get_work(chunk_size: u32) -> Vec<&'static str> {
    CEF_SAMPLE[0..chunk_size as usize].to_vec()
}

pub fn get_device_count() -> usize {
    let mut devices = vec![];
    let platforms = Platform::list();
    for platform in platforms {
        match core::get_device_ids(&platform, Some(ocl::flags::DEVICE_TYPE_GPU), None) {
            Ok(device_ids) => {
                devices = [devices, device_ids].concat();
            }
            Err(_) => {}
        };
    }
    devices.len()
}

fn flatten_strings(ss: impl Iterator<Item = &'static str>) -> Vec<u8> {
    let mut res = Vec::new();
    for s in ss {
        res.extend(s.as_bytes());
        res.extend(['\n' as u8]);
    }
    res
}
