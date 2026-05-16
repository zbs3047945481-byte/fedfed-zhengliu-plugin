from docx import Document
files=['tmp/docs/reference/1-ref.docx','/Users/zhubingshuo/毕设资料/2-复制.docx']
for fp in files:
    d=Document(fp)
    print('FILE', fp)
    shown=0
    for i,p in enumerate(d.paragraphs):
        t=p.text.strip()
        if not t:
            continue
        if t.startswith('5.') or len(t)>30:
            pf=p.paragraph_format
            r=p.runs[0] if p.runs else None
            font=r.font if r else None
            print(i, repr(t[:55]), 'style', p.style.name, 'size', font.size.pt if font and font.size else None, 'font', font.name if font else None, 'line', pf.line_spacing, 'rule', pf.line_spacing_rule, 'first', pf.first_line_indent.pt if pf.first_line_indent else None, 'before', pf.space_before.pt if pf.space_before else None, 'after', pf.space_after.pt if pf.space_after else None)
            shown+=1
            if shown>=20:
                break
