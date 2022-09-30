#[macro_use]
extern crate lazy_static;

use num_format::{Locale, ToFormattedString};
use ocl::builders::ContextProperties;
use ocl::prm::cl_uchar;
use ocl::{core, flags, Platform};
use rand::Rng;
use rayon::prelude::*;
use std::ffi::CString;
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

    println!("Found {} GPU devices, will use only 1 for this POC, compiling kernel...", devices.len());
    let src = r#"
                __kernel void kernel_cefparser(__global const uchar * input, __global uchar * output) {
                    ulong idx = get_global_id(0);
                
                    uchar prev_char_to_inspect = input[idx-1];
                    uchar char_to_inspect = input[idx];
                    uchar output_classification = 0;
                
                    if (char_to_inspect == '=' && prev_char_to_inspect != '\\') {
                        output_classification = 1;
                    }
                    else if (char_to_inspect == ' ') {
                    output_classification = 4;
                    }
                    else if (char_to_inspect == '\n') {
                    output_classification = 5;
                    }
                    else if (char_to_inspect == '|') {
                    output_classification = 2;
                    }
                    output[idx] = output_classification;
                }
            "#;
    let src_cstring = CString::new(src).unwrap();

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
                .arg_named("input", None::<&ocl::Buffer<cl_uchar>>)
                .arg_named("output", None::<&ocl::Buffer<cl_uchar>>)
                .build()
                .unwrap();

            let mut chunk_size = 1_000_000;
            let mut counter = 0;
            let mut bytes = 0;

            loop {
                let work_stdin = get_work(chunk_size as u32);
                let mut _chunk_size = work_stdin.len();
                if _chunk_size > 0 {
                    let input = flatten_strings(work_stdin.clone().into_iter());
                    let input_buf = ocl::Buffer::builder()
                        .context(&context)
                        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
                        .copy_host_slice(&input)
                        .len(input.len())
                        .build()
                        .unwrap();
                    let mut output: Vec<cl_uchar> = vec![0 as cl_uchar; input.len()];
                    let output_buf = ocl::Buffer::builder()
                        .context(&context)
                        .flags(flags::MEM_WRITE_ONLY | flags::MEM_COPY_HOST_PTR)
                        .copy_host_slice(&output)
                        .len(output.len())
                        .build()
                        .unwrap();
                    chunk_size = work_stdin.len() as u128;

                    kernel.set_arg("input", &input_buf).unwrap();
                    kernel.set_arg("output", &output_buf).unwrap();

                    unsafe {
                        kernel
                            .cmd()
                            .global_work_size(
                                ocl::SpatialDims::new(Some(input.len() as usize), None, None)
                                    .unwrap(),
                            )
                            .queue(&queue)
                            .enq()
                            .unwrap();
                        output_buf.read(&mut output).queue(&queue).enq().unwrap();
                        let mut msg_idx = 0;
                        for msg in output.split(|x| *x == 5) {
                            if msg.len() == 0 {
                                continue;
                            }

                            let pos: Vec<usize> = msg
                                .iter()
                                .enumerate()
                                .filter_map(|(idx, val)| if *val == 1 { Some(idx) } else { None })
                                .collect();

                            let mut json = "{".to_string();
                            for (i, p) in pos.iter().enumerate() {
                                let mut start = 0;
                                let mut end = 0;
                                for i in (0..*p).rev().step_by(1) {
                                    if msg[i] == 4 || msg[i] == 2 {
                                        start = i + 1;
                                        break;
                                    }
                                }
                                if i + 1 >= pos.len() {
                                } else {
                                    let last = pos[i + 1];
                                    for i in (*p..last).rev() {
                                        if msg[i] == 4 {
                                            end = i;
                                            break;
                                        }
                                    }

                                    json.push_str("\"");
                                    json.push_str(&work_stdin[msg_idx][start..*p]);
                                    json.push_str("\":");
                                    json.push_str("\"");
                                    json.push_str(&work_stdin[msg_idx][*p + 1..end]);
                                    json.push_str("\",");
                                }
                            }
                            json.truncate(json.len() - 1);
                            json.push('}');
                            //println!("{}", json);
                            msg_idx += 1;
                        }
                    }

                    let duration = start.elapsed();
                    total_counter += chunk_size;
                    total_bytes += input.len() as f32 / 1024.0 / 1024.0 / 1024.0;

                    counter += chunk_size;
                    bytes += input.len();

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
        "<134>2022-02-14T03:17:30-08:00 TEST CEF:0|Vendor|Product|20.0.560|600|User Signed In|3|src=127.0.0.1 suser=Admin target=Admin msg=User signed in from 127.0.0.1 Tenant=Primary TenantId=0 act=",
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
        res.extend(s.chars().map(|x| x as u8));
        res.extend(['\n' as u8]);
    }
    res
}
