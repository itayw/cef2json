# cef2json

This is a Proof-of-Concept (POC) for parsing ArcSight CEF securtiy log records with the assistance of GPUs.

TLDR; an initial result of **x30 more events per second using GPUs** for log streaming/processing of CEF events.

CEF is very similar to the basic `syslog` format, the main goal of the parser is to turn the raw textual data into a better data format for future handling, such as json. there are a few differences to the syslog/CEF standards, so normal syslog parsers cannot be used for this task. There are several parsers out there as part of data shipping suites, such as: Logstash, fluentd, Splunk collectors and you can also find open source libraries for the purpose of CEF parsing. 

You can learn more about the CEF format [here](https://community.microfocus.com/cyberres/productdocs/w/connector-documentation/38809/arcsight-common-event-format-cef-implementation-standard). 

### Motivation

I've been working around CEF for almost 10 years now and written/used several parsers over time. Usually, CEF parsing is done in order to push the security logs into external systems, such as databases (Elasticsearch) or SIEM products (Microsoft Sentinel). So, the parsing part is just a small part of the whole process, however it's a very CPU/memory consuming stage which heavily effects scaling and costs. 

This is the focus of this POC, scaling the CEF parser to be able to deal with more events for less money.

Over the last couple of years I learned Rust and OpenCL, I wanted to experiment with the two and see if we can leverage their capabilities and concepts for faster and cheaper CEF parsing.

### How does it work?

The two traditional approaches for CEF parsing are:
- Regex - this is very expensive and will yield poor scaling results.
- Loops - this is the common solution, we iterate over the CEF event string and look for specific chars to indicate how we should parse the string.

My approach was different, I wanted to avoid the loops and make the work parallel, this will allow us to harness the GPU.

The main algorhythm is as follows:
- Read CEF messages from buffer (can be file, stdin, memory, kafka)
- Align all messages on a single vector of uchar
- Pass vector to GPU
- Each thread checks if the vector[idx] is a special char
- An output vector is used to specify for each char if it's special
- The output vector is broken back to individual messages
- Based on the output vector we identify for each `=` the nearest left/right `[space]`
- Lastly, we iterate over the positions and output a concatenated string of json.

> Special chars are `=`, `\n`, `|` and `[space]`

### Running the code

```
$ cargo run --release
```

### TODO

There are many more things to do, this is an incomplete parser, it does not comply with all required standards and cannot be used outside of this POC.

- [ ] Standard compliance with RFC
- [ ] Additional input/output modes
- [ ] SIMD 
- [ ] Optimization
- [ ] Better handling of host/device (svm)

### Results

Initial results of the main concept show a **x30 improvement** over the current CEF parsers available in the market.

```
Found 2 GPU devices, will use only 1 for this POC, compiling kernel...
device DeviceId(0x556e1d7dd0e0) loading
Processed: 3,000,000 events/0.59548753 gb, EPS 771760.56 events/second, GBPS 0.15319127 gb/second
Processed: 6,000,000 events/1.1909751 gb, EPS 817725.56 events/second, GBPS 0.16231512 gb/second
Processed: 9,000,000 events/1.7864627 gb, EPS 812648.9 events/second, GBPS 0.16130742 gb/second
Processed: 12,000,000 events/2.3819501 gb, EPS 807267.9 events/second, GBPS 0.16023932 gb/second
Processed: 15,000,000 events/2.9774377 gb, EPS 809192.25 events/second, GBPS 0.1606213 gb/second
Processed: 18,000,000 events/3.5729253 gb, EPS 829809.1 events/second, GBPS 0.16471367 gb/second
Processed: 21,000,000 events/4.1684127 gb, EPS 834042.6 events/second, GBPS 0.165554 gb/second
Processed: 24,000,000 events/4.7639003 gb, EPS 835721.5 events/second, GBPS 0.16588725 gb/second
Processed: 27,000,000 events/5.359388 gb, EPS 837910.94 events/second, GBPS 0.16632184 gb/second
Processed: 30,000,000 events/5.9548755 gb, EPS 833865.94 events/second, GBPS 0.16551892 gb/second
```

Hardware used:

```
Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
Nvidia GeForce 1070 Ti
```