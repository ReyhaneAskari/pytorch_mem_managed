Credit goes to: https://github.com/prigoyal/pytorch_memonger

This repo contains the tests for using cpu as swap for gpu in pytorch. The changes in pytorch are in this branch: https://github.com/ReyhaneAskari/pytorch/tree/manage_cpu_ram


Benchmark results:


|                                | RESNET             | RESNET managed     | RESNET managed      | RESNET managed     
|--------------------------------|--------------------|--------------------|---------------------|--------------------
| Total iterations (runs * iter) | 4 * 10             | 4 * 10             | 2 * 10              | 1 * 10             
| mini-batch-size                | 8                  | 8                  | 16                  | 32                 
| avg gpu time over 10 iters     | 1.581939 s         | 8.094114 s         | 5.041900 s          | 7.056532 s         
| total time except 1st run      | 14.2369 s          | 72.846 s           | 45.105              | 62.662 s           
| total time                     | 27.171s            | 88.398s            | 58.478s             | 78.065s            
| Data on GPU                    | 8120MiB / 12066MiB | 8172MiB / 12066MiB | 12064MiB / 12066MiB | 12064MiB / 12066MiB





|                                | VNET               | VNET managed       | VNET managed        | VNET managed        
|--------------------------------|--------------------|--------------------|---------------------|---------------------
| Total iterations (runs * iter) | 4 * 10             | 4 * 10             | 2 * 10              | 1 * 10              
| mini-batch-size                | 4                  | 4                  | 8                   | 16                  
| avg gpu time over 10 iters     | 10.536860 s        | 11.024819 s        | 15.294250 s         | 17.153561 s         
| total time except 1st run      | 94.8305            | 99.222 s           | 137.144 s           | 136.810 s           
| total time                     | 121.88827 s        | 130.893s           | 178.042 s           | 216.938 s           
| Data on GPU                    | 9452MiB / 12066MiB | 8622MiB / 12066MiB | 12064MiB / 12066MiB | 12064MiB / 12066MiB 



|                                | Word Language model            | Word Language model managed   | Word Language model managed     | Word Language model managed     | Word Language model            | Word Language model managed    |
|--------------------------------|--------------------|-------------------|---------------------|---------------------|--------------------|--------------------|
| Total iterations (runs * iter) | 5 * 20             | 5 * 20            | 1 * 20              | 1 * 20              | 5 * 20             | 5 * 20             |
| mini-batch-size                | 50                 | 50                | 250                 | 50                  | 250                | 250                |
| bptt (sequence length)         | 200                | 200               | 200                 | 1000                | 40                 | 40                 |
| avg gpu time over 10 iters     | 0.4911             | 0.59014 s         | 8.9360 s            | 9.282 s             | 0.3872             | 0.3906 s           |
| total time except 1st run      | 9.220              | 11.1005 s         | 168.912 s           | 175.477 s           | 7.2463             | 7.310 s            |
| total time                     | 20.438s            | 16.193s           | 181.989s            | 188.707s            | 11.528s            | 12.290 s           |
| Data on GPU                    | 7484MiB / 12066MiB | 7550MiB / 12066Mi | 12064MiB / 12066MiB | 12064MiB / 12066MiB | 7486MiB / 12066MiB | 7552MiB / 12066MiB |
