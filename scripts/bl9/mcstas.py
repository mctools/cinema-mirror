#!/usr/bin/env python3
import os
import json

def read_EELL(fname='section.json', ifprint=False):
    fname = os.path.join(os.path.dirname(__file__), fname)
    with open(fname, 'r') as f:
        eell = json.load(f)
    if ifprint:
        print(eell)
    return eell.values()

eell = read_EELL()
output_file = 'guide_channels.instr'  

# 定义需要创建的 guide_channel 数量  
num_channels = len(eell)
eell = list(eell)
# 打开输出文件  
with open(output_file, 'w') as f:  
    # 写入文件头  
    f.write("DEFINE INSTRUMENT GuideChannels(origin_location = 0)\n\n")  
    f.write("TRACE\n\n")  
    f.write("COMPONENT origin = Progress_bar() AT (0,0,origin_location) ABSOLUTE\n\n")

    # write source
    f.write(f"COMPONENT Source = Source_gen(\n")
    f.write(f"dist = 5, \n")
    f.write(f"focus_xw = 0.06, \n")
    f.write(f"focus_yh = 0.06, \n")
    f.write(f"lambda0 = 3, \n")
    f.write(f"dlambda = 2.9,\n")
    f.write(f"yheight = 0.02, \n")
    f.write(f"xwidth = 0.02,\n")
    f.write(f"T1=293) \n")
    f.write(f"AT (0, 0, 8.2165-5) RELATIVE PREVIOUS\n\n")

    # 循环生成 guide_channel  
    for i in range(num_channels):  
        w1 = eell[i]['entryOpening']['halfwidth'] * 2 * 0.001
        h1 = eell[i]['entryOpening']['halfheight'] * 2 * 0.001
        w2 = eell[i]['exitOpening']['halfwidth'] * 2 * 0.001
        h2 = eell[i]['exitOpening']['halfheight'] * 2 * 0.001
        ll = eell[i]['length'] * 0.001
        location = eell[i]['zlocation'] * 0.001

        # 定义 guide_channel 的位置和参数  
        f.write(f"COMPONENT Guide_section{i+1} = Guide_channeled(\n")  
        f.write(f"w1={w1},\n")  
        f.write(f"h1={h1},\n")  
        f.write(f"w2={w2},\n")  
        f.write(f"h2={h2},\n")  
        f.write(f"l={ll},\n")  
        f.write(f"R0={0.99},\n")  
        f.write(f"Qc={0.0219},\n")  
        f.write(f"alpha={6.07},\n")  
        f.write(f"mx={2},\n")  
        f.write(f"my={2}\n")  
        f.write(")\n")  
        f.write(f"AT (0, 0, {location}) RELATIVE  origin\n\n")
    
    # Monitor
    f.write(f"COMPONENT Monitor1 = Monitor_nD(\n")
    f.write(f'options = "x bins=100 y bins=100",\n')
    f.write(f"xwidth = 0.1,\n")
    f.write(f"yheight = 0.1\n")
    f.write(")\n")  
    f.write(f"AT (0, 0, {location+ll+2}) RELATIVE  origin\n\n")


    # 文件尾部  
    f.write("END\n")  

print(f"生成 {num_channels} 个 guide_channel 的脚本已保存为 {output_file}.")  