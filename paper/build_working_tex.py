"""Build a clearly marked review draft from current prose and audited R figures."""
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parent
CITATIONS={
    'https://arxiv.org/html/2503.20783v2':'liu2025drgrpo',
    'https://proceedings.mlr.press/v306/li26dw.html':'li2026gr3',
    'https://openreview.net/pdf?id=GP5RHZnEsw':'li2025drpo',
    'https://arxiv.org/html/2503.14476v2':'yu2025dapo',
    'https://arxiv.org/html/2601.02780v1':'xiaomi2026mimo',
    'https://arxiv.org/abs/2402.03300v3':'shao2024deepseekmath',
    'https://arxiv.org/abs/2103.03874v2':'hendrycks2021math',
    'https://arxiv.org/abs/2110.14168v2':'cobbe2021gsm8k',
    'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct':'qwen25model',
    'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507':'qwen3model',
    'https://huggingface.co/openbmb/MiniCPM5-2B':'minicpm5model',
    'https://arxiv.org/abs/2106.09685v2':'hu2021lora',
    'https://www.jstor.org/stable/4615733':'holm1979',
    'https://doi.org/10.2202/1544-6115.1585':'phipson2010',
}


def escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                  '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(char,char) for char in text)


def inline(text):
    pattern=r'\[([^\]]+)\]\((https?://[^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`|(\$\$.*?\$\$|\$[^$\n]+\$)'
    output=[];last=0
    for match in re.finditer(pattern,text,re.S):
        output.append(escape(text[last:match.start()]))
        if match.group(1):
            if match.group(2) in CITATIONS:output.append(escape(match.group(1))+r'~\cite{'+CITATIONS[match.group(2)]+'}')
            else:output.append(r'\href{'+match.group(2)+r'}{'+escape(match.group(1))+'}')
        elif match.group(3):output.append(r'\textbf{'+escape(match.group(3))+'}')
        elif match.group(4):output.append(r'\texttt{'+escape(match.group(4))+'}')
        else:
            formula=match.group(5)
            output.append(r'\['+formula[2:-2].strip()+r'\]' if formula.startswith('$$') else formula)
        last=match.end()
    output.append(escape(text[last:]))
    return ''.join(output)


def bibliography(cited_keys):
    # Deliberately limited to the one-line braced fields in this verified local file.
    # Fail on unfamiliar syntax instead of silently dropping bibliographic metadata.
    source=(ROOT/'references_verified.bib').read_text(encoding='utf-8')
    entries={}
    for kind,key,body in re.findall(r'^@(\w+)\{(\w+),\n(.*?)^\}',source,re.M|re.S):
        fields={}
        for line in body.split('\n'):
            if not line.strip():continue
            match=re.fullmatch(r'\s*(\w+)\s*=\s*\{(.*)\},?\s*',line)
            if not match:raise ValueError(f'Unsupported bibliography field: {line}')
            fields[match.group(1)]=match.group(2)
        assert key not in entries and {'author','title','url'}<=fields.keys()
        entries[key]=fields
    assert set(cited_keys)<=entries.keys()
    output=[r'\small',r'\begin{thebibliography}{99}']
    for key in cited_keys:
        entry=entries[key];authors=[]
        for name in entry['author'].split(' and '):
            if ',' in name:
                surname,given=name.split(',',1);name=given.strip()+' '+surname.strip()
            authors.append(name.replace('{','').replace('}',''))
        title=entry['title'].replace('{','').replace('}','')
        publication=entry.get('booktitle',entry.get('journal',entry.get('howpublished','')))
        if 'volume' in entry:publication+=', vol. '+entry['volume']
        if 'pages' in entry:publication+=', pp. '+entry['pages']
        if 'year' in entry:publication+=', '+entry['year']
        note=entry.get('note','')
        output.append(r'\bibitem{'+key+'} '+escape(', '.join(authors))+'. '+
            r'\emph{'+escape(title)+'}. '+escape(publication)+'. '+
            (escape(note)+'. ' if note else '')+r'\url{'+entry['url']+'}.')
    output.append(r'\end{thebibliography}')
    return '\n\n'.join(output)


def main():
    previous=[ROOT/name for name in ('manuscript_working.tex','manuscript_working.pdf') if (ROOT/name).exists()]
    if previous:
        archive=ROOT/'draft_archive'/str(time.time_ns());archive.mkdir(parents=True)
        for path in previous:(archive/path.name).write_bytes(path.read_bytes())
    source=(ROOT/'MANUSCRIPT_WORKING.md').read_text(encoding='utf-8')
    title=source.split('\n',1)[0].removeprefix('# ')
    body=[]
    for paragraph in source.split('\n\n')[1:]:
        paragraph=paragraph.strip()
        if not paragraph:continue
        if paragraph.startswith('**Wang Yueran**'):continue
        if paragraph.startswith('## '):
            heading=paragraph[3:]
            if heading=='Conclusion': body.append(r'\clearpage')
            if heading=='Appendix: Historical Audit and Exploratory Protocol': body.append(r'\appendix')
            body.append(r'\section{'+escape(heading)+'}')
            if heading=='Reward Construction and Analysis':
                body.append(r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\textwidth]{figure_candidates/candidate_method_steps.pdf}'+'\n'+r'\caption{Training and evaluation with an illustrative recorded group. Panel A summarizes the workflow; the listed completion lengths and labels come from Qwen2.5 centered shaping, seed 42, step 36, task mathtr\_tr\_141, selected as the first recorded three-success group without truncation. Panel B shows its recorded rewards and normalized advantages. This example illustrates one group rather than the distribution of all groups. The quality-only arm is absent from the MiniCPM extension.}\label{fig:method}\end{figure}')
            if heading=='Results':
                body.append(r'\input{results_absolute_table.tex}')
                body.append(r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\textwidth]{figure_candidates/candidate_primary_grouped.pdf}'+'\n'+r'\caption{Paired accuracy differences for centered shaping versus the stated comparator. Open circles show each of three seed effects; diamonds show their mean. Thin dark intervals are conditional item bootstrap 95\% intervals; broad pale intervals are crossed seed-item bootstrap 95\% intervals. Blue denotes the four Qwen primary MATH comparisons; teal denotes the two MiniCPM extension comparisons. They form separate prespecified multiplicity families. Crossed intervals are approximate with three seeds.}\label{fig:primary}\end{figure}')
                body.append(r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\textwidth]{figure_candidates/candidate_group_mechanism.pdf}'+'\n'+r'\caption{Observed success-count composition in centered-shaping training runs. Panel a reports the pooled percentage of 600 recorded groups per model across three seeds; horizontal offsets separate model symbols. Panel b shows the three seed-specific percentages for groups with exactly three successes, with a diamond for their mean. The group-level condition identifies a possible direct distinction between centering rules on the same sampled group; it does not identify a causal explanation for downstream accuracy.}\end{figure}')
            if heading=='Discussion and Limitations':
                body.append(r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\textwidth]{figures_v3/fig05_accuracy_by_cap.pdf}'+'\n'+r'\caption{Strict accuracy by Qwen model, dataset, reward arm and output cap. Open circles show all three seeds; horizontal offsets distinguish arms without changing their cap values. Short budgets are prefixes, not independently timed runs.}\end{figure}')
                body.append(r'\begin{figure}[!htbp]\centering'+'\n'+r'\includegraphics[width=\textwidth]{figure_candidates/candidate_budget_paired.pdf}'+'\n'+r'\caption{Descriptive centered-minus-binary strict accuracy differences across output caps. Each open symbol represents one matched training seed; pale lines connect that seed across saved prefixes and diamonds show the three-seed mean. All panels use a common vertical scale. Short caps reuse prefixes and are not independently timed generations. Only the Qwen MATH 1,024-token comparison belongs to the primary test family; the other cells are descriptive.}\end{figure}')
        else:body.append(inline(paragraph.replace('\n',' '))+'\n')
    captions=(ROOT/'FIGURE_CAPTIONS.md').read_text(encoding='utf-8')
    figure_captions=re.findall(r'\*\*Figure \d+\. (.*?)\*\* (.*?)(?=\n\n|\Z)',captions,re.S)
    assert len(figure_captions)>=3
    figure_captions=figure_captions[:3]
    body.append(r'\clearpage')
    files=['fig01_accuracy','fig02_signal_cost','fig03_protocol_pilot']
    for index,(name,(heading,caption)) in enumerate(zip(files,figure_captions)):
        if name=='fig02_signal_cost':
            caption=caption.replace('Point shapes follow Figure 1.', 'Point shapes follow the preceding historical-accuracy figure.')
        if name=='fig03_protocol_pilot':
            body.append(r'\begin{figure}[p]\centering')
            for mi,model,penalty in [(1,'Qwen2.5-1.5B-Instruct','1.1'),(2,'Qwen3-4B-Instruct-2507','1.0')]:
                detail=('Exploratory protocol comparison for '+model+'. The left panel shows paired accuracy differences on the same 16 development questions under strict and historical scoring; intervals are unadjusted 95% paired-item bootstrap intervals from 20,000 resamples. Free denotes the unconstrained prompt. The right panel connects output lengths for the same question under 256- and 512-token caps; circles denote EOS and triangles the length limit. Horizontal offsets separate overlapping observations without changing token counts. These are development observations, not independent confirmation. The pilot repetition penalty was '+penalty+'; between-model contrasts cannot isolate model identity or size.')
                if mi==2: detail='The same exploratory comparison as the preceding figure for Qwen3-4B-Instruct-2507; the pilot repetition penalty was 1.0. Differences between models do not isolate model identity or size.'
                body.append(r'\includegraphics[width=\textwidth]{figure_candidates/candidate_pilot_model_'+str(mi)+r'.pdf}'+'\n'+r'\caption{'+inline(detail)+r'}')
            body.append(r'\end{figure}')
            continue
        body.append((r'\begin{figure}[p]\centering' if index in (0,2) else r'\vspace{14pt}')+ '\n'+
            r'\includegraphics[width=\textwidth,height=.60\textheight,keepaspectratio]{figures_audit_20261017/'+name+r'.pdf}'+ '\n'+
            r'\caption{\textbf{'+inline(heading)+'} '+inline(caption)+'}'+
            (r'\end{figure}\clearpage' if index in (1,2) else ''))
    preamble=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=24mm]{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[colorlinks=true,urlcolor=blue]{hyperref}
\pagestyle{plain}
\setlength{\headheight}{14pt}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\emergencystretch=2em
'''
    body_text='\n\n'.join(body)
    cited_keys=list(dict.fromkeys(re.findall(r'\\cite\{(\w+)\}',body_text)))
    tex=preamble+'\\title{'+escape(title)+'}\n\\author{Wang Yueran\\textsuperscript{*}\\\\School of Artificial Intelligence, Beijing Normal University\\\\Correspondence: \\texttt{wangyueran@mail.bnu.edu.cn}}\n\\date{}\n\\begin{document}\n\\maketitle\n'+body_text+'\n\\clearpage\n'+bibliography(cited_keys)+'\n\\end{document}\n'
    (ROOT/'manuscript_working.tex').write_text(tex,encoding='utf-8')


if __name__=='__main__':main()


