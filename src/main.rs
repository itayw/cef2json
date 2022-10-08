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
    let chunk_size = 10000;
    let mut start = Instant::now();

    let mut total_counter = 0;
    let mut total_bytes = 0.0;
    let mut counter = 0;
    let mut bytes = 0;

    let mut work_size = 0;

    loop {
        let work_stdin = get_work(chunk_size as u32);
        let _chunk_size = work_stdin.len();

        if _chunk_size > 0 {
            work_stdin.par_iter().for_each(|work| {
                let work = *work;
                //println!("work {}", work);

                let idx = work
                    .chars()
                    .enumerate()
                    .map(|(i, x)| {
                        if x == '|' && work.chars().nth(i - 1).unwrap() != '\\' {
                            1
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<usize>>();

                //split the header away
                let idx = work
                    .chars()
                    .enumerate()
                    .filter_map(|(i, x)| {
                        if x == '|' && work.chars().nth(i - 1).unwrap() != '\\' {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<usize>>();
                let cef_start = idx[idx.len() - 1];
                let header = &work[..cef_start];
                let cef_message = &work[cef_start + 1..];

                let idx = cef_message
                    .chars()
                    .enumerate()
                    .filter_map(|(i, x)| {
                        if x == '=' && work.chars().nth(i - 1).unwrap() != '\\' {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<usize>>();
                //println!("{:?}::::{:?}", idx,cef_message);

                //parse header
            });
        }
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
