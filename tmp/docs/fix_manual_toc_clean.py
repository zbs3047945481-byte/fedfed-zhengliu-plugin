from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml.ns import qn

path = '/Users/zhubingshuo/毕设资料/2-复制.docx'
doc = Document(path)
entries = [
    ('toc 1', '摘  要', 'I'), ('toc 1', 'Abstract', 'II'),
    ('toc 1', '第1章 绪  论', '1'), ('toc 2', '1.1 课题背景概述', '1'), ('toc 2', '1.2 现有解决方法', '1'), ('toc 2', '1.3 局限性', '2'),
    ('toc 1', '第2章 相关理论与方法基础', '3'), ('toc 2', '2.1 联邦学习', '3'), ('toc 2', '2.2 FedAvg', '3'), ('toc 2', '2.3 FedProx', '3'), ('toc 2', '2.4 FedNova', '4'), ('toc 2', '2.4 SCAFFOLD', '4'), ('toc 2', '2.5 数据蒸馏与特征分离思想', '4'), ('toc 2', '2.7 FedFed', '5'),
    ('toc 1', '第3章 系统总体设计与数据异构的实现', '6'), ('toc 2', '3.1 系统设计目标', '6'), ('toc 2', '3.2 系统总体架构与插件化设计', '6'), ('toc 2', '3.3 本章小结', '6'),
    ('toc 1', '第4章 FedFed特征蒸馏插件设计与实现', '7'), ('toc 2', '4.1 FedFed插件模块组成', '7'), ('toc 2', '4.2 图像空间特征蒸馏方法', '7'), ('toc 2', '4.3 两阶段插件训练机制', '11'), ('toc 2', '4.4 共享样本池设计与辅助信息交互', '11'), ('toc 2', '4.5 本章小结', '13'),
    ('toc 1', '第5章 实验设计与性能评估', '14'), ('toc 2', '5.1 实验目的', '14'), ('toc 2', '5.2 实验环境与基础设置', '14'), ('toc 2', '5.3 评价指标与公平性控制', '14'), ('toc 2', '5.4 FedFed 插件主性能验证', '15'), ('toc 2', '5.5 主性能改进历程分析', '18'), ('toc 2', '5.6 插件关键模块消融实验', '18'), ('toc 2', '5.7 本章小结', '20'),
    ('toc 1', '第6章 机制验证与工程分析', '22'), ('toc 2', '6.1 特征蒸馏机制诊断', '22'), ('toc 2', '6.2 隐私验证', '23'), ('toc 2', '6.3 跨联邦优化器可插拔验证', '28'), ('toc 2', '6.4 训练开销分析', '30'), ('toc 2', '6.5 本章小结', '31'),
    ('toc 1', '结  论', '32'), ('toc 1', '参考文献', '33'), ('toc 1', '致  谢', '35'),
]
paras = doc.paragraphs
start = next(i for i,p in enumerate(paras) if p.text.strip() == '目  录') + 1
end = next(i for i,p in enumerate(paras) if p.style.name == 'Heading 1' and p.text.strip().startswith('第1章'))
for p in list(doc.paragraphs[start:end]):
    p._element.getparent().remove(p._element)
anchor = next(p for p in doc.paragraphs if p.style.name == 'Heading 1' and p.text.strip().startswith('第1章'))
# Insert before the first chapter; insert in natural order because each new paragraph becomes the new previous sibling.
for style, title, page in entries:
    p = anchor.insert_paragraph_before(f'{title}\t{page}', style=style)
    pf = p.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = Pt(16)
    try:
        pf.tab_stops.clear_all(); pf.tab_stops.add_tab_stop(Pt(410), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    except Exception:
        pass
    for run in p.runs:
        run.font.name = '宋体'; run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体'); run.font.size = Pt(10.5)
for p in doc.paragraphs:
    if p.text.strip() == '第5章 实验设计与性能评估':
        p.paragraph_format.page_break_before = True
doc.save(path)
print('cleaned and rebuilt TOC')
