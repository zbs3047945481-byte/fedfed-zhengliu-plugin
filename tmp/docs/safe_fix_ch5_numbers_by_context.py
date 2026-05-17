from docx import Document
DOC='/Users/zhubingshuo/毕设资料/2-复制.docx'
d=Document(DOC)
section=None
rules={
 '5.4.1 主性能对比': {'表 6-2':'表 5-1','图 6-2':'图 5-1','图 6-3':'图 5-2'},
 '5.4.2 数据异构强度分析': {'表 6-3':'表 5-2','图 6-4':'图 5-3','图 6-5':'图 5-4'},
 '5.4.3 本地训练强度分析': {'表 6-2':'表 5-3','图 6-2':'图 5-5'},
 '5.6.1 共享特征池规模消融': {'表 6-3':'表 5-4','图 6-3':'图 5-6'},
 '5.6.2 特征蒸馏充分性消融': {'表 6-2':'表 5-5','图 6-4':'图 5-7'},
}
for p in d.paragraphs:
    t=p.text.strip()
    if t in rules:
        section=t; continue
    if t.startswith('5.') and t not in rules and p.style.name.startswith('Heading'):
        section=None
    if t.startswith('第6章'):
        section=None
    repl=rules.get(section)
    if not repl or not any(old in p.text for old in repl):
        continue
    for r in p.runs:
        if 'w:drawing' in r._r.xml or 'w:pict' in r._r.xml: continue
        txt=r.text
        for old,new in repl.items(): txt=txt.replace(old,new)
        r.text=txt
d.save(DOC)
