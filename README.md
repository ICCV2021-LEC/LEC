# LEC-Net
Anonymous code for ICCV 2021 "Learnable Expansion-and-Compression Network for Few-shot Class-Incremental Learning", paper id 2389

## Overview
- `config/` contains the config file;
- `data/` contains the dataloader and dataset including CIFAR100, CUB200 and miniImageNet;
- `data_list/` contains the data list to index data;
- `models/` contains the implementation of the LECNet(`LECNet.py`) and the Resnet backbone file;
- `snapshots/` contains the pretrained LECNet parameters;
- `utils/` contains other dependency;

<table style="center">
    <caption align="center">Performance comparison on CUB with Resnet18 backbone</caption>
    <tr>
        <td rowspan=2>Method</td> 
        <td colspan="11" align="center">Sessions</td> 
   </tr>
    <tr>
        <td>0</td> 
        <td>1</td> 
        <td>2</td> 
        <td>3</td> 
        <td>4</td> 
        <td>5</td> 
        <td>6</td> 
        <td>7</td> 
        <td>8</td> 
        <td>9</td> 
        <td>10</td> 
    </tr>
    <tr>
        <td>LEC-Net</td> 
        <td>70.86</td> 
        <td>58.15</td> 
        <td>54.83</td> 
        <td>49.34</td> 
        <td>45.85</td> 
        <td>40.55</td> 
        <td>39.70</td> 
        <td>34.59</td> 
        <td>36.58</td> 
        <td>33.56</td> 
        <td>31.96</td> 
    </tr>
</table>

<table align="center">
    <caption align="center">Performance comparison on CIFAR100 with Resnet18 backbone</caption>
    <tr>
        <td rowspan=2>Method</td> 
        <td colspan="9" align="center">Sessions</td> 
   </tr>
    <tr>
        <td>0</td> 
        <td>1</td> 
        <td>2</td> 
        <td>3</td> 
        <td>4</td> 
        <td>5</td> 
        <td>6</td> 
        <td>7</td> 
        <td>8</td> 
    </tr>
    <tr>
        <td>LEC-Net</td> 
        <td>64.10</td> 
        <td>53.23</td> 
        <td>44.19</td> 
        <td>41.87</td> 
        <td>38.54</td> 
        <td>39.54</td> 
        <td>37.34</td> 
        <td>34.73</td> 
        <td>34.73</td> 
    </tr>
</table>
