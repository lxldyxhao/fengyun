# encoding: utf-8

# 处理  var125-126

import pandas as pd
import re

# 区号
code_df = pd.read_csv("../../data/area_code.csv", encoding='gbk')
code_dict = {}
for i in range(0, len(code_df)):
    code_dict.update({code_df['区号'][i]: code_df['城市'][i]})
# 省名
provinces = ('河北', '山西', '辽宁', '吉林', '江苏', '浙江', '安徽', '福建',
             '江西', '山东', '河南', '湖北', '湖南', '广东', '海南', '四川',
             '贵州', '云南', '陕西', '甘肃', '青海', '广西', '西藏', '宁夏',
             '新疆', '黑龙江', '内蒙古')

def repl_in(s):
    while (True):
        search_res = re.search(r'@([0-9]\d*)@', s)
        if search_res == None:
            return s
        num_str = search_res.group().replace('@', '')
        num = int(num_str)
        if num in code_dict.keys():
            s = s.replace(num_str, code_dict[num])
        else:
            s = s.replace(num_str, '未知')

def repl_left(s):
    while (True):
        search_res = re.search(r'^([0-9]\d*)@', s)
        if search_res == None:
            return s
        num_str = search_res.group().replace('@', '')
        num = int(num_str)
        if num in code_dict.keys():
            s = s.replace(num_str, code_dict[num])
        else:
            s = s.replace(num_str, '未知')

def repl_right(s):
    while (True):
        search_res = re.search(r'@([0-9]\d*)$', s)
        if search_res == None:
            return s
        num_str = search_res.group().replace('@', '')
        num = int(num_str)
        if num in code_dict.keys():
            s = s.replace(num_str, code_dict[num])
        else:
            s = s.replace(num_str, '未知')

def repl_only(s):
    while (True):
        search_res = re.search(r'^([0-9]\d*)$', s)
        if search_res == None:
            return s
        num_str = search_res.group().replace('@', '')
        num = int(num_str)
        if num in code_dict.keys():
            s = s.replace(num_str, code_dict[num])
        else:
            s = s.replace(num_str, '未知')

def repl_sheng(s):
    if s == None:
        return ''
    if len(s) >= 4:
        if (s[:2] in provinces):
            return s.replace(s[:2], '')[:2] + ' '
        elif (s[:3] in provinces):
            return s.replace(s[:3], '')[:2] + ' '
        else:
            return s[:2] + ' '
    elif len(s) == 3:
        return s[:2] + ' '
    return s + ' '

# 处理var125,var126
def process(var):
    var = var.fillna('空值')
    var = var.str.strip().str.replace('[A-Z]*','')\
        .str.replace('[a-z]*','').str.replace('\[|\]','')
    var = var.str.strip().str.replace('省|市','').str.replace('+','')\
        .str.replace('；','').str.replace('.','')
    var = var.apply(repl_in).apply(repl_left).apply(repl_right).apply(repl_only)
    var = var.str.replace('[0-9]*','')
    var = var.str.replace(' 公司开通3G出访','').str.replace('（|）','')\
        .str.replace('成都资阳眉山三地','成都').str.replace('--','未知')
    var = var.str.split('@', expand=True)
    for col in var:
        var[col] = var[col].apply(repl_sheng)

    return var.T.sum().str.strip()


