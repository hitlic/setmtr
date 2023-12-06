"""
统计满足如下条件的化学式中每个原子的数量。
    - 水合物 ·3H2O替换为 (H2O)3
    - 不能有电荷上脚标
    - 不能有“-”或“=”符号
    - 不能有“[”和“]”
    - 不能是Co(acac)2
"""

from bs4 import BeautifulSoup


def load_html_file(file_name='无机化合物CAS号列表.html'):
    """
    加载html文件，返回化合物分子式、名称、CAS号
    """
    with open(file_name, encoding='utf-8') as f:
        page = f.read()
    soup = BeautifulSoup(page, features="html.parser")

    data_lst = []
    for i, tb in enumerate(soup.find_all('table', class_="wikitable")):
        for tr in tb.find_all('tr')[1:]:
            formula, name, cas = tr.find_all("td")
            contents = [str(e).replace('<sub>','').replace('</sub>', '') for e in formula.contents]
            if '<br/>' in contents:
                idx = contents.index('<br/>')
                contents = [contents[:idx], contents[idx+1:]]
            else:
                contents = [contents]
            for content in contents:
                data_lst.append([preprocess(''.join(content)), name.text, cas.text.strip()])
    return data_lst

def preprocess(f):
    """
    预处理代学式，去除方括号，将水合物中 ·3H2O替换为 (H2O)3
    """
    f = f.replace('[', '').replace(']', '')
    if '·' in f:
        f = f[:-3].replace("·", "(H2O)")
    return f

def cut_segs(f):
    """
    将化学式中的化学元素切分出来，对于化学式中的“()”使用了递归
    """
    segs = []

    seg = ''
    num = ''
    i = 0
    while i < len(f):
        c = f[i]
        if c.isupper():         # 大写符号
            if seg != '':
                segs.append([seg, '1' if num == '' else num])
            seg = c
            num = ''
            i += 1
        elif c.islower():       # 小写符号
            seg += c
            i += 1
        elif c.isnumeric():     # 数字
            num += c
            i += 1
        elif c == '(':          # 括号
            if seg != '':
                segs.append([seg, '1' if num == '' else num])
                num = ''
            idx = f.index(')', i)
            sub_f = f[i+1:idx]

            # 正确匹配括号
            beg_idx = i
            while '(' in sub_f:
                idx = f.index(')', idx+1)
                beg_idx = f.index('(', beg_idx+1)
                sub_f = f[beg_idx+1:idx]

            seg = cut_segs(f[i+1:idx])
            i = idx + 1
    # 保存最后一个元素信息
    segs.append([seg, '1' if num == '' else num])
    return segs

def count(segs):
    eles = {}
    for seg, num in segs:
        if isinstance(seg, str):
            if seg in eles:
                eles[seg] += int(num)
            else:
                eles[seg] = int(num)
        else:
            for s, n in count(seg).items():
                if s in eles:
                    eles[s] += n * int(num)
                else:
                    eles[s] = n * int(num)
    return eles

def decompose(formula):
    segs = cut_segs(formula)
    return count(segs)



if __name__ == "__main__":
    # t = 'ZnCH2Co13(C8H15O)2(H2O)4'
    # t = 'Cu(C18H33O2)2'
    # t = 'Co3(Fe(CN)6)2'
    # print(decompose(t))

    data = load_html_file()


    lines1 = set()
    lines2 = []
    for formula, name, cas in data:
        componds = decompose(formula)
        line1 = ','.join(componds.keys())
        line2 = f'{cas}: ' + ','.join([f'{k}@[{v}]' for k, v in componds.items()])
        lines1.add(line1+'\n')
        lines2.append(line2+"\n")
    with open('inorganic_compound.txt', "w", encoding='utf-8') as f1:
        f1.writelines(lines1)
    with open('inorganic_compound_with_atom_num.txt', "w", encoding='utf-8') as f2:
        f2.writelines(lines2)
