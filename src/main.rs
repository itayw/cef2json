#[macro_use]
extern crate lazy_static;

use crossbeam_channel::bounded;
use crossbeam_queue::ArrayQueue;
use num_format::{Locale, ToFormattedString};
use ocl::builders::ContextProperties;
use ocl::core::Uint2;
use ocl::prm::{cl_uchar, cl_uint, Uint3};
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

    let devices:Vec<ocl::core::DeviceId> = devices.iter().map(|x| *x).take(1).collect();
    println!(
        "Found {} GPU devices, will use only 1 for this POC, compiling kernel...",
        devices.len()
    );
    let src = r#"
        __kernel void kernel_parser(__global uchar * input, __global uint * lengths, __global uint * starts, __global uint3 * output, __global uint * pair_counts) {
            ulong idx = get_global_id(0);
            
            uint start = starts[idx];
            uint length = lengths[idx];
            uint end = start + length;
            uint key_block = 50;

            uint pair_count = 0;
            uint pair_start = 0;
            uint pair_end = end;
            uint pair_eq = 0;
            bool looking_for_pair_start = false;

            for (int i = end; i > start ; i--) {
                if (input[i] == '=' && input[i-1] != '\\') {
                    //we have an equal sign, let's look for a the complete keypair
                    pair_eq = i;
                    looking_for_pair_start = 1;
                }
                if (looking_for_pair_start && (input[i] == ' ' || input[i] == '|')) {
                    pair_start = i + 1;

                    //we have a keypair, let's push it
                    uint pos = (key_block * idx) + pair_count;

                    uint3 result = {pair_start, pair_end, pair_eq};
                    output[pos] = result;
                    
                    //init our vars for the next lookup
                    looking_for_pair_start = false;
                    pair_end = i;
                    pair_start = 0;
                    pair_eq = 0;

                    pair_count++;
                }
            }

            pair_counts[idx] = pair_count;
        }
    "#;
    let src_cstring = CString::new(src).unwrap();

    //devices
       // .into_par_iter()
        //.skip(1)
        //.take(1)
        //.for_each(move |device_id| {
            let device_id = devices[0];
            let mut start = Instant::now();

            let mut total_counter = 0;
            let mut total_bytes: f32 = 0.0;

            let chunk_size = 50000;
            let mut counter = 0;
            let mut bytes = 0;
            let key_block = 50;

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

            let mut output = vec![Uint3::default(); chunk_size * key_block];
            let output_buf = unsafe {
                ocl::Buffer::builder()
                    .context(&context)
                    .flags(flags::MEM_WRITE_ONLY)
                    .use_host_slice(&output)
                    .len(output.len())
                    .build()
                    .unwrap()
            };
            //println!("output {}ms", iteration_start.elapsed().as_millis());
            //let mut pairs_count = vec![0 as cl_uint; chunk_size];
            let pairs_count_buf = unsafe {
                ocl::Buffer::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_WRITE_ONLY)
                    .fill_val(0 as cl_uint)
                    .len(chunk_size)
                    .build()
                    .unwrap()
            };

            let kernel = ocl::Kernel::builder()
                .program(&program)
                .name("kernel_parser")
                .arg_named("input", None::<&ocl::Buffer<cl_uchar>>)
                .arg_named("lengths", None::<&ocl::Buffer<cl_uint>>)
                .arg_named("starts", None::<&ocl::Buffer<cl_uint>>)
                .arg_named("output", &output_buf)
                .arg_named("pairs_count", &pairs_count_buf)
                .build()
                .unwrap();

            loop {
                let work_stdin = get_work(chunk_size as u32);
                let _chunk_size = work_stdin.len();
                //let mut work_size = 0;
                //let mut first = true;

                if _chunk_size > 0 {
                    let mut iteration_start = Instant::now();
                    let (mut input, (lengths, starts)) =
                        flatten_strings(work_stdin.clone().into_iter());
                    //println!("flatten {}ms", iteration_start.elapsed().as_millis());
                    unsafe {
                        let input_buf = ocl::Buffer::builder()
                            .context(&context)
                            .flags(flags::MEM_READ_ONLY)
                            .use_host_slice(&input)
                            .len(input.len())
                            .build()
                            .unwrap();
                        //println!("input {}ms", iteration_start.elapsed().as_millis());
                        let lengths_buf = ocl::Buffer::builder()
                            .context(&context)
                            .flags(flags::MEM_READ_ONLY)
                            .use_host_slice(&lengths)
                            .len(lengths.len())
                            .build()
                            .unwrap();
                        //println!("lengths {}ms", iteration_start.elapsed().as_millis());
                        let starts_buf = ocl::Buffer::builder()
                            .context(&context)
                            .flags(flags::MEM_READ_ONLY)
                            .use_host_slice(&starts)
                            .len(starts.len())
                            .build()
                            .unwrap();
                        //println!("starts {}ms", iteration_start.elapsed().as_millis());
                        //work_size = input.len();

                        //println!("prep {}ms", iteration_start.elapsed().as_millis());
                        kernel.set_arg("input", &input_buf).unwrap();
                        kernel.set_arg("starts", &starts_buf).unwrap();
                        kernel.set_arg("lengths", &lengths_buf).unwrap();

                        //println!("args {}ms", iteration_start.elapsed().as_millis());

                        kernel
                            .cmd()
                            .global_work_size(
                                ocl::SpatialDims::new(Some(_chunk_size as usize), None, None)
                                    .unwrap(),
                            )
                            //.local_work_size(kernel.default_local_work_size())
                            .queue(&queue)
                            .enq()
                            .unwrap();

                        output_buf.read(&mut output).queue(&queue).enq().unwrap();
                        /*pairs_count_buf
                        .read(&mut pairs_count)
                        .queue(&queue)
                        .enq()
                        .unwrap();*/

                        //output
                        //    .par_chunks(key_block)
                        //    .enumerate()
                        //    .for_each(|(idx, chunk)| {
                            for (idx,chunk) in output.chunks(key_block).enumerate(){
                                let mut chunk = chunk.to_vec();
                                chunk.retain(|x| x[1] > 0);
                                let headers = &input
                                    [starts[idx] as usize..chunk[chunk.len() - 1][0] as usize]
                                    .split(|x| *x == '|' as u8)
                                    .collect::<Vec<&[u8]>>();
                                //println!("headers {:?}", headers);

                                for pair in chunk {
                                    let key = &input[pair[0] as usize..pair[2] as usize];
                                    let start = pair[2] + 1;
                                    let end = {
                                        if start > pair[1] {
                                            start
                                        } else {
                                            pair[1]
                                        }
                                    };
                                    let value = &input[start as usize..end as usize];
                                    //}
                                    //println!("{:?}:{:?}", key, value);
                                    /*println!(
                                        "{:?}:{:?}",
                                        key.iter().map(|x| *x as char).collect::<String>(),
                                        value.iter().map(|x| *x as char).collect::<String>()
                                    );*/
                                }
                            }
                        //    });
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
        //});
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

#[inline(always)]
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

#[inline(always)]
fn flatten_strings(ss: impl Iterator<Item = &'static str>) -> (Vec<u8>, (Vec<u32>, Vec<u32>)) {
    let mut res = vec![];
    let mut lengths = vec![];
    let mut starts = vec![];

    let mut start = 0;
    for s in ss {
        let s = s.as_bytes();
        res.extend(s);
        //res.push('\n' as u8);
        lengths.push(s.len() as u32);
        starts.push(start as u32);
        start += s.len();
    }
    (res, (lengths, starts))
}
