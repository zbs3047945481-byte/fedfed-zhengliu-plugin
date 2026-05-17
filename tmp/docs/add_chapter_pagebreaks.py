from docx import Document
from docx.enum.text import WD_BREAK
DOC='/Users/zhubingshuo/毕设资料/2-复制.docx'
d=Document(DOC)
for p in d.paragraphs:
    if p.text.strip() in {'第6章 机制验证与工程分析','结  论'}:
        prev=p._p.getprevious()
        # Avoid duplicate: if previous paragraph already has a page break, skip.
        need=True
        if prev is not None and 'lastRenderedPageBreak' in prev.xml:
            need=False
        brp=d.add_paragraph()
        brp.paragraph_format.space_before=0
        brp.paragraph_format.space_after=0
        brp.add_run().add_break(WD_BREAK.PAGE)
        p._p.addprevious(brp._p)

d.save(DOC)
