#![feature(portable_simd)]
#[macro_use]
extern crate lazy_static;

use num_format::{Locale, ToFormattedString};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::{self, Context};
use opencl3::device::{Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
//use opencl3::error_codes::cl_int;
use opencl3::types::{cl_event, cl_kernel, cl_uchar, cl_uint};
use std::ptr;

use libc::{c_void, size_t};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{CL_MAP_READ, CL_MAP_WRITE};
use opencl3::program::{Program, CL_STD_2_0,CL_STD_3_0};
use opencl3::svm::SvmVec;

use opencl3::types::CL_BLOCKING;
use opencl3::Result;
use rand::Rng;
use rayon::prelude::*;
use std::ffi::CString;
use std::time::{Duration, Instant};

const STAT_INTERVAL: u64 = 3;

fn main() {
    let mut devices = vec![];
    for platforms in opencl3::platform::get_platforms() {
        for platform in platforms {
            let _devices = platform.get_devices(CL_DEVICE_TYPE_CPU).unwrap();
            for device in _devices {
                devices.push(Device::new(device));
            }
        }
    }

    println!("Found {} GPU devices, compiling kernel...", devices.len());
    let src = r#"
        __kernel void kernel_parser(__global const uchar * input, __global const uint * lengths, __global const uint * starts, __global uint4 * output, __global uint * pair_counts) {
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

                    uint4 result = {pair_start, pair_end, pair_eq, idx};
                    *(output + pos) = result;
                    //output[pos] = result;
                    
                    //init our vars for the next lookup
                    looking_for_pair_start = false;
                    pair_end = i;
                    pair_start = 0;
                    pair_eq = 0;

                    pair_count++;
                }
            }

            *(pair_counts + idx ) = pair_count;
            //pair_counts[idx] = pair_count;
        }
    "#;
    //let src_cstring = CString::new(src).unwrap();

    (0..8).into_par_iter().for_each(move |thread_idx| {
        let device = devices[0];
        let mut start = Instant::now();

        let mut total_counter = 0;
        let mut total_bytes: f32 = 0.0;

        let chunk_size = 50_000;
        let mut counter = 0;
        let mut bytes = 0;
        let key_block = 50;

        println!("device {:?}:{} loading", device, thread_idx);
        let context = Context::from_device(&device).expect("Context::from_device failed");

        let queue = unsafe {
            CommandQueue::create_with_properties(
                &context,
                context.default_device(),
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .expect("CommandQueue::create_default_with_properties failed")
        };
        let program = unsafe {
            Program::create_and_build_from_source(&context, &src, CL_STD_2_0)
                .expect("Program::create_and_build_from_source failed")
        };

        //println!("output {}ms", iteration_start.elapsed().as_millis());
        //let mut pairs_count = vec![0 as cl_uint; chunk_size];
        /*let pairs_count_buf = ocl::Buffer::builder()
        .queue(queue.clone())
        .flags(flags::MEM_WRITE_ONLY)
        .fill_val(0 as cl_uint)
        .len(chunk_size)
        .build()
        .unwrap();*/

        let kernel =
            Kernel::create(&program, "kernel_parser").expect("Kernel::create failed") ;
            let mut output =
                    SvmVec::<std::simd::u32x4>::allocate_zeroed(&context, chunk_size * key_block)
                        .expect("SVM allocation failed");
                let mut pairs_count = SvmVec::<cl_uint>::allocate_zeroed(&context, chunk_size)
                    .expect("SVM allocation failed");

        loop {
            let work_stdin = get_work(chunk_size as u32);
            let _chunk_size = work_stdin.len();
            if _chunk_size > 0 {
                let (input, (lengths, starts)) = flatten_strings(work_stdin.clone().into_iter());
                // Copy into an OpenCL SVM vector
                let mut input_buf = SvmVec::<cl_uchar>::allocate(&context, input.len())
                    .expect("SVM allocation failed");
                input_buf.copy_from_slice(&input);
                let input = input_buf;
                //println!("input {:?}", input);

                let mut lengths_buf = SvmVec::<cl_uint>::allocate(&context, lengths.len())
                    .expect("SVM allocation failed");
                lengths_buf.copy_from_slice(&lengths);
                let lengths = lengths_buf;

                let mut starts_buf = SvmVec::<cl_uint>::allocate(&context, starts.len())
                    .expect("SVM allocation failed");
                starts_buf.copy_from_slice(&starts);
                let starts = starts_buf;
    
                let kernel_event = unsafe {
                    ExecuteKernel::new(&kernel)
                        .set_arg_svm(input.as_ptr())
                        .set_arg_svm(lengths.as_ptr())
                        .set_arg_svm(starts.as_ptr())
                        .set_arg_svm(output.as_mut_ptr())
                        .set_arg_svm(pairs_count.as_mut_ptr())
                        .set_global_work_size(_chunk_size)
                        .enqueue_nd_range(&queue)
                        .unwrap()
                };

                /*
                unsafe {
                    //kernel.set_arg("input", &input_buf).unwrap();
                    //kernel.set_arg("starts", &starts_buf).unwrap();
                    //kernel.set_arg("lengths", &lengths_buf).unwrap();

                    kernel
                        .set_arg_svm_pointer(0, input.as_ptr() as *const c_void)
                        .expect("Failed to set kernel SVM pointer");
                    kernel
                        .set_arg_svm_pointer(1, starts.as_ptr() as *const c_void)
                        .expect("Failed to set kernel SVM pointer");
                    kernel
                        .set_arg_svm_pointer(2, lengths.as_ptr() as *const c_void)
                        .expect("Failed to set kernel SVM pointer");
                }
                unsafe {
                    kernel
                        .set_arg_svm_pointer(3, output.as_mut_ptr() as *const c_void)
                        .expect("Failed to set kernel SVM pointer");

                    kernel
                        .set_arg_svm_pointer(4, pairs_count.as_mut_ptr() as *const c_void)
                        .expect("Failed to set kernel SVM pointer");
                }
                println!("input len {}", input.len());
                let mut event_wait_list: Vec<cl_event> = vec![];
                let kernel_event = unsafe {
                    queue
                        .enqueue_nd_range_kernel(
                            kernel.get(),
                            1,
                            ptr::null(),
                            [input.len() as size_t].as_ptr(),
                            ptr::null(),
                            &mut event_wait_list,
                        )
                        .unwrap()
                };
                */
                kernel_event.wait().unwrap();

                //println!("output {:?}", output);
                //println!("pairs_count {:?}", pairs_count);

                for chunk in output.chunks(key_block) {
                    let mut chunk = chunk.to_vec();
                    chunk.retain(|x| x[1] > 0);
                    let _headers = &input
                        [starts[chunk[0][3] as usize] as usize..chunk[chunk.len() - 1][0] as usize]
                        .split(|x| *x == '|' as u8)
                        .collect::<Vec<&[u8]>>();
                    //println!("headers {:?}", headers);

                    for pair in chunk {
                        let _key = &input[pair[0] as usize..pair[2] as usize];
                        let start = pair[2] + 1;
                        let end = {
                            if start > pair[1] {
                                start
                            } else {
                                pair[1]
                            }
                        };
                        let _value = &input[start as usize..end as usize];
                        //}
                        //println!("{:?}:{:?}", key, value);
                        /*println!(
                            "{}, {:?}:{:?}",
                            pair[3],
                            _key.iter().map(|x| *x as char).collect::<String>(),
                            _value.iter().map(|x| *x as char).collect::<String>()
                        );*/
                    }
                }

                /*
                               unsafe {
                                   kernel
                                       .cmd()
                                       .global_work_size(
                                           ocl::SpatialDims::new(Some(_chunk_size as usize), None, None).unwrap(),
                                       )
                                       //.local_work_size(kernel.default_local_work_size())
                                       .queue(&queue)
                                       .enq()
                                       .unwrap();
                               }

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
                               for chunk in output.chunks(key_block) {
                                   let mut chunk = chunk.to_vec();
                                   chunk.retain(|x| x[1] > 0);
                                   let _headers = &input
                                       [starts[chunk[0][3] as usize] as usize..chunk[chunk.len() - 1][0] as usize]
                                       .split(|x| *x == '|' as u8)
                                       .collect::<Vec<&[u8]>>();
                                   //println!("headers {:?}", headers);

                                   for pair in chunk {
                                       let _key = &input[pair[0] as usize..pair[2] as usize];
                                       let start = pair[2] + 1;
                                       let end = {
                                           if start > pair[1] {
                                               start
                                           } else {
                                               pair[1]
                                           }
                                       };
                                       let _value = &input[start as usize..end as usize];
                                       //}
                                       //println!("{:?}:{:?}", key, value);
                                       /*println!(
                                           "{}, {:?}:{:?}",
                                           pair[3],
                                           _key.iter().map(|x| *x as char).collect::<String>(),
                                           _value.iter().map(|x| *x as char).collect::<String>()
                                       );*/
                                   }
                               }
                               //    });
                */
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
