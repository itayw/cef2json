#[macro_use]
extern crate lazy_static;

use crossbeam_channel::bounded;
use crossbeam_queue::ArrayQueue;
use num_format::{Locale, ToFormattedString};
use ocl::builders::ContextProperties;
use ocl::core::Uint2;
use ocl::prm::{cl_uchar, cl_uint};
use ocl::{core, flags, Platform};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::hash::Hash;
use std::time::{Duration, Instant};

const STAT_INTERVAL: u64 = 3;

struct Positions {
    start: usize,
    end: usize,
    idx: usize,
}

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
                #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

                int increase(__global int *addrCounter, int size) {
                    return atomic_inc(addrCounter);
                }

                __kernel void kernel_indexes(__global uchar * in_out, __global uint * ln_count, __global uint * ln, __global uint * pairs_count, __global uint * pairs) {
                    ulong idx = get_global_id(0);

                    if (in_out[idx] == '=' && in_out[idx - 1] != '\\') {
                        int fakePointer = increase(&pairs_count[0], 1); // serial operation
                        pairs[fakePointer] = idx;
                    }
                    /*else if (in_out[idx] == '\n') {
                        int fakePointer = increase(&ln_count[0], 1); // serial operation
                        ln[fakePointer] = idx;
                    }*/
                }

                __kernel void kernel_parts(__global uchar * in_out, __global uint * pairs, __global uint2 * pairs2) {
                    ulong idx = get_global_id(0);
                    ulong pair_idx = pairs[idx];
                    uint distance = 0;
                    for (int i = pair_idx ; i > pair_idx - 32 ; i--) {
                        if (in_out[i] == ' ' || in_out[i] == '|') {
                            pairs2[idx].s0 = pair_idx - distance + 1;
                            if (idx - 1 > 0){
                                pairs2[idx-1].s1 = pair_idx - distance;
                            }
                            break;
                        }
                        
                        distance++;
                    }
                }                
            "#;
    let src_cstring = CString::new(src).unwrap();

    /*
    //let q = ArrayQueue::new(1_000_000);
    let s: crossbeam_channel::Sender<(Vec<u8>, Vec<u8>)>;
    let r: crossbeam_channel::Receiver<(Vec<u8>, Vec<u8>)>;
    (s, r) = bounded(10_000_000);

    std::thread::spawn(move || {
        for _ in 1..8 {
            loop {
                let (in_out, work_stdin) = r.recv().unwrap();

                //println!("{:?}", r.recv().unwrap());
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    });*/

    devices
        .into_par_iter()
        //.skip(1)
        .take(1)
        .for_each(move |device_id| {
            let mut start = Instant::now();

            let mut total_counter = 0;
            let mut total_bytes: f32 = 0.0;

            let chunk_size = 25000;
            let mut counter = 0;
            let mut bytes = 0;

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

            let mut ln = vec![0 as cl_uint; chunk_size as usize]; //((_chunk_size as f64 * 0.0625) + 1.0)
            let ln_buf = unsafe {
                ocl::Buffer::builder()
                    .context(&context)
                    .flags(flags::MEM_WRITE_ONLY)
                    .use_host_slice(&ln)
                    .len(ln.len())
                    .build()
                    .unwrap()
            };

            let mut pairs = vec![0 as cl_uint; chunk_size * 256 as usize]; //((_chunk_size as f64 * 0.0625) + 1.0)
            let pairs_buf = unsafe {
                ocl::Buffer::builder()
                    .context(&context)
                    .flags(flags::MEM_WRITE_ONLY)
                    .use_host_slice(&pairs)
                    .len(pairs.len())
                    .build()
                    .unwrap()
            };

            let kernel = ocl::Kernel::builder()
                .program(&program)
                .name("kernel_indexes")
                .arg_named("in_out", None::<&ocl::Buffer<cl_uchar>>)
                .arg_named("ln_count", None::<&ocl::Buffer<cl_uint>>)
                .arg_named("ln", &ln_buf)
                .arg_named("pairs_count", None::<&ocl::Buffer<cl_uint>>)
                .arg_named("pairs", &pairs_buf)
                .build()
                .unwrap();

            let kernel_parts = ocl::Kernel::builder()
                .program(&program)
                .name("kernel_parts")
                .arg_named("in_out", None::<&ocl::Buffer<cl_uchar>>)
                .arg_named("pairs", &pairs_buf)
                .arg_named("pairs2", None::<&ocl::Buffer<Uint2>>)
                .build()
                .unwrap();
            loop {
                let work_stdin = get_work(chunk_size as u32);
                let _chunk_size = work_stdin.len();
                let mut work_size = 0;
                let mut first = true;
                if _chunk_size > 0 {
                    let mut in_out = flatten_strings(work_stdin.clone().into_iter());
                    unsafe {
                        let in_out_buf = ocl::Buffer::builder()
                            .context(&context)
                            .flags(flags::MEM_READ_WRITE)
                            .use_host_slice(&in_out)
                            //.copy_host_slice(&in_out)
                            .len(in_out.len())
                            .build()
                            .unwrap();

                        let mut ln_count = vec![0 as cl_uint; 1 as usize];
                        let ln_count_buf = unsafe {
                            ocl::Buffer::builder()
                                .context(&context)
                                .flags(flags::MEM_WRITE_ONLY)
                                .use_host_slice(&ln_count)
                                .len(ln_count.len())
                                .build()
                                .unwrap()
                        };

                        let mut pairs_count = vec![0 as cl_uint; 1 as usize];
                        let pairs_count_buf = unsafe {
                            ocl::Buffer::builder()
                                .context(&context)
                                .flags(flags::MEM_WRITE_ONLY)
                                .use_host_slice(&pairs_count)
                                .len(pairs_count.len())
                                .build()
                                .unwrap()
                        };

                        work_size = in_out.len();
                        let _input = in_out.clone();

                        kernel.set_arg("in_out", &in_out_buf).unwrap();
                        kernel.set_arg("ln_count", &ln_count_buf).unwrap();
                        kernel.set_arg("pairs_count", &pairs_count_buf).unwrap();

                        kernel
                            .cmd()
                            .global_work_size(
                                ocl::SpatialDims::new(Some(work_size as usize), None, None)
                                    .unwrap(),
                            )
                            /* .local_work_size(
                                ocl::SpatialDims::new(Some(32 as usize), Some(1 as usize), None)
                                    .unwrap(),
                            )*/
                            .queue(&queue)
                            .enq()
                            .unwrap();

                        //in_out_buf.read(&mut in_out).queue(&queue).enq().unwrap();
                        //ln_count_buf.read(&mut ln_count).queue(&queue).enq().unwrap();
                        ln_buf.read(&mut ln).queue(&queue).enq().unwrap();
                        pairs_count_buf
                            .read(&mut pairs_count)
                            .queue(&queue)
                            .enq()
                            .unwrap();
                        pairs_buf.read(&mut pairs).queue(&queue).enq().unwrap();

                        let mut pairs = pairs[0..pairs_count[0] as usize].to_vec();
                        pairs.sort();
                        let pairs_buf = unsafe {
                            ocl::Buffer::builder()
                                .context(&context)
                                .flags(flags::MEM_WRITE_ONLY)
                                .use_host_slice(&pairs)
                                .len(pairs.len())
                                .build()
                                .unwrap()
                        };

                        let mut pairs2 = vec![Uint2::default(); pairs_count[0] as usize];
                        let pairs2_buf = unsafe {
                            ocl::Buffer::builder()
                                .context(&context)
                                .flags(flags::MEM_WRITE_ONLY)
                                .use_host_slice(&pairs2)
                                .len(pairs2.len())
                                .build()
                                .unwrap()
                        };

                        kernel_parts.set_arg("in_out", &in_out_buf).unwrap();
                        kernel_parts.set_arg("pairs", &pairs_buf).unwrap();
                        kernel_parts.set_arg("pairs2", &pairs2_buf).unwrap();

                        kernel_parts
                            .cmd()
                            .global_work_size(
                                ocl::SpatialDims::new(Some(pairs_count[0] as usize), None, None)
                                    .unwrap(),
                            )
                            /* .local_work_size(
                                ocl::SpatialDims::new(Some(32 as usize), Some(1 as usize), None)
                                    .unwrap(),
                            )*/
                            .queue(&queue)
                            .enq()
                            .unwrap();

                        //in_out_buf.read(&mut in_out).queue(&queue).enq().unwrap();
                        //ln_count_buf.read(&mut ln_count).queue(&queue).enq().unwrap();

                        pairs2_buf.read(&mut pairs2).queue(&queue).enq().unwrap();

                        let pair = &_input[pairs2[5][0] as usize..pairs2[5][1] as usize];

                        //println!("pair {:?}",  pair.iter().map(|x| *x as char).collect::<String>());

                        //

                        /*
                        ln.sort();
                        //println!("ln {:?}", ln);
                        let mut pairs = pairs[0..pairs_count[0] as usize].to_vec();
                        //let mut pairs:Vec<&u32> = pairs.iter().filter(|x| *x > &0).collect();
                        pairs.sort();

                        let mut last_pair_idx = 0;
                        let mut pair_indexes = vec![];
                        for pair_idx in pairs {
                            let mut msg_idx = 0;
                            for i in &ln {
                                //println!("compoare {} {}", pair_idx, i);
                                if pair_idx as u32 >= *i {
                                    break;
                                }
                                msg_idx += 1;
                            }

                            let pos = Positions {
                                start: last_pair_idx,
                                end: pair_idx as usize,
                                idx: msg_idx,
                            };

                            pair_indexes.push(pos);

                            last_pair_idx = pair_idx as usize + 1;
                        }

                        pair_indexes.par_iter().for_each(|pos| {
                            /*let pair = _input[*start..*end]
                                .to_vec()
                                .iter()
                                .map(|x| *x as char)
                                .collect::<String>();
                            println!("pair {}", pair);*/

                            let pair = &_input[pos.start..pos.end];

                            //println!("msg {}, pair {:?}", pos.idx, pair);
                        });*/
                        //println!("pairs {} ", pairs.iter().filter(|x| *x > &0).count());

                        //std::process::exit(1);
                        /*let line_breaks: Vec<usize> = in_out
                        .iter()
                        .enumerate()
                        .filter_map(|(i, x)| if *x == 1 { Some(i) } else { None })
                        .collect();&*/
                        //println!("line_breaks {:?}", line_breaks);

                        /*

                        let line_breaks=ln;
                        let mut last_ln = 0;
                        for ln in line_breaks {
                            let indexes = &in_out[last_ln..ln as usize];
                            let line = &_input[last_ln..ln as usize];

                            match s.send((indexes.to_vec(), line.to_vec())) {
                                Ok(()) => {}
                                Err(e) => {
                                    println!("Error sending");
                                }
                            }
                            /*println!(
                                "line {}",
                                line.iter().map(|x| *x as char).collect::<String>()
                            );*/
                            //println!("indexes {:?}", indexes);
                            /*
                                                        let part_breaks: Vec<usize> = indexes
                                                            .iter()
                                                            .enumerate()
                                                            .filter_map(|(i, x)| if *x == 16 { Some(i) } else { None })
                                                            .collect();

                                                        let mut last_pa = 0;
                                                        for pa in part_breaks {
                                                            let pa_indexes = &indexes[last_pa..pa];
                                                            let part = &line[last_pa..pa];

                                                            let pos = pa_indexes
                                                                .iter()
                                                                .enumerate()
                                                                .filter_map(|(i, x)| if *x == 8 { Some(i) } else { None })
                                                                .collect::<Vec<usize>>();
                                                            if pos.len() > 0 {
                                                                let pos = pos[0];
                                                                let key = &part[..pos];
                                                                let value = &part[pos..];
                            //let key  =String::from_utf8(key.to_vec()).unwrap();
                            //let value = String::from_utf8(value.to_vec()).unwrap() ;
                                                                /*println!(
                                                                    "kv {}:{}",
                                                                    String::from_utf8(key.to_vec()).unwrap(),
                                                                    String::from_utf8(value.to_vec()).unwrap() //part.iter().map(|x| *x as char).collect::<String>()
                                                                );*/
                                                            }
                                                            //println!("pa_indexes {:?}", pa_indexes);

                                                            last_pa = pa + 1;
                                                        }
                            */
                            //println!("part_breaks {:?}", part_breaks);

                            last_ln = ln as usize + 1;
                        }*/

                        //in_out.par_split(|x| *x == 1).for_each(|line|{

                        //for part in line.split(|x| *x == 16)  {
                        //let keypair = part.split(|x|*x == 8).map(|x| x.to_vec()).collect::<Vec<Vec<u8>>>();
                        //println!("keypair {:?}",keypair);
                        //let key = keypair[0].iter().map(|x| *x as char).collect::<String>();
                        //let value = keypair[1].iter().map(|x| *x as char).collect::<String>();

                        //}

                        //  });

                        let mut last_ln = 0;
                        let mut last_part = 0;
                        //let mut last = vec![];
                        //let mut last_idx = vec![];
                        //let mut msg = vec![];
                        //let mut part = vec![];
                        //println!("in {:?}", in_out);
                        /*for i in 0..in_out.len() {

                            let current = in_out[i].clone();
                            if current == 1 {
                                //println!("last {}", last.iter().map(|x| *x as char).collect::<String>());
                                //println!("last_idx {:?}", last_idx);

                                let pos:Vec<usize> = last_idx.iter().enumerate().filter_map(|(i,x)| if *x ==16 { Some (i)} else {None}).collect();
                                //println!("pos {:?}",pos);
                                //we reached the end of the message
                                //msg.push(part.clone());
                                //part.clear();
                                last_ln = i;

                            } else {
                                last.push(_input[i]);
                                last_idx.push(in_out[i]);
                            }
                        }*/
                    }
                    /*_input.par_split(|x|*x=='\n' as u8).for_each(|pos| {
                        if pos.len() == 0 {
                            return;
                        }
                        //let msg = work_stdin[idx];
                        //println!("msg {:?}", msg);
                        //println!("pos {:?}", pos);


                    }); */

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
