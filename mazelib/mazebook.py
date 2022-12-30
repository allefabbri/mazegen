from fpdf import FPDF
from glob import glob
import pandas as pd
from PIL import Image

class PDF(FPDF):

    def header(self):
        self.set_font('Arial', 'B', 20)
        self.set_text_color(0,0,0)
        self.cell(80)
        self.cell(30, 10, 'The great book of mazes', 0, 0, 'C')
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(0,0,0)
        l = self.add_link()
        self.cell(0, 10, f'Page {self.page_no()} - Return to contents', 0, 0, 'C', link=l)
        self.set_link(l, page=0)

    def chapter_title(self, title):
        self.set_font('Arial', '', 17)
        self.cell(30, 10, title)
        self.ln()

    def chapter_text(self, title, cell_h=10, fontsize=12, fontstyle=''):
        self.set_font('Arial', fontstyle, fontsize)
        self.cell(30, cell_h, title)
        self.ln()

def compute_imsize(imgpath, pdf, h_thresh=0.9, v_thresh=0.8):
    im = Image.open(imgpath)
    iw, ih = im.size
    image_w = h_thresh * pdf.w
    image_h = image_w * ih / iw
    if image_h > v_thresh * pdf.h:
        image_h = v_thresh * pdf.h
        image_w = image_h * iw / ih
    return image_w, image_h

def make_book(mazedir):
    mazefiles = glob(f'{mazedir}/*.png')
    data = [ f.split('-') + [f] for f in mazefiles ]
    data = [
        {
            'type': d[1],
            'size': d[2],
            'id': int(d[3]),
            'mode': d[-2].split('.')[0],
            'file': d[-1],
        } for d in data
    ]
    df = pd.DataFrame(data)
    df = df.sort_values(by=['type', 'size', 'id'])
    #print(df)

    pdflinks = {}
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_text(f'Contents', fontstyle='B', fontsize=17)

    for mtype, dfg in df.groupby('type'):
        pdf.cell(5)
        pdf.chapter_text(f'\tSection {mtype}', cell_h=7, fontsize=15)
        for msize, dfg1 in dfg.groupby('size'):
            pdf.cell(10)
            pdf.chapter_text(f'\t\tSize {msize}', cell_h=7, fontsize=13)
            for mid, _ in dfg1.groupby('id'):
                utag = f'{mtype}-{msize}-{mid}'
                pdflinks[utag] = pdf.add_link()
                pdf.set_text_color(0,0,255)
                pdf.set_font('Arial', 'U', 11)
                pdf.cell(15)
                pdf.cell(40, 7, f'Maze {mid:04d}', border=False, align='', fill=False, link=pdflinks[utag])
                pdf.ln()
                pdf.set_text_color(0,0,0)

    for nmaze, (tags, dg) in enumerate(df.groupby(['type', 'size', 'id'])):
        # print(tags)
        mtype, msize, mid = tags
        utag = f'{mtype}-{msize}-{mid}'

        if len(dg) != 3:
            print(f'Error with tag : {tags}')
            continue

        mazeimg = dg[ dg['mode'] == 'labyrinth' ].file.values[0]
        dbgimg = dg[ dg['mode'] == 'debug' ].file.values[0]
        solimg = dg[ dg['mode'] == 'solved' ].file.values[0]
        print(mazeimg)

        pdf.add_page()
        pdf.set_link(pdflinks[utag], pdf.page_no())
        pdf.chapter_title(f'Maze #{nmaze+1}')
        pdf.chapter_text(f'Type {tags[0]} size {tags[1]} id {tags[2]}')

        wm,hm = compute_imsize(mazeimg, pdf)
        offset_w = (pdf.w - wm) / 2
        pdf.image(mazeimg, offset_w, 50, wm, hm)

        pdf.add_page()
        pdf.chapter_title(f'Maze #{nmaze+1} - Solution and Debug')

        w,h = compute_imsize(solimg, pdf, v_thresh=0.4)
        offset_w1 = (pdf.w - w) / 2
        pdf.image(solimg, offset_w1, 40, w, h)

        w,h = compute_imsize(dbgimg, pdf, v_thresh=0.4)
        offset_w1 = (pdf.w - w) / 2
        pdf.image(dbgimg, offset_w1, 40 + h, w, h)

    pdf.output(f'{mazedir}/mazebook.pdf', 'F')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--book', action='store_true')
    parser.add_argument('-d', '--dir', required=True)
    args = parser.parse_args()

    make_book(args.dir)
