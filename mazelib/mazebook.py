# from fpdf import FPDF

# pdf = FPDF()


# # create a cell
# pdf.add_page()
# pdf.set_font("Arial", size = 15)
# pdf.cell(200, 10, txt = 'Alle is Alle')

# pdf.add_page()
# pdf.set_font("Arial", size = 25)
# pdf.cell(200, 10, txt = 'Alle is Alle')

# # save the pdf with name .pdf
# pdf.output("mazebook.pdf")

from fpdf import FPDF
from glob import glob
import pandas as pd

class PDF(FPDF):

    def header(self):
        # Logo
        # Arial bold 15
        self.set_font('Arial', 'B', 20)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'The great book of mazes', 0, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', '', 17)
        self.cell(30, 10, title)#, 1, 0, 'C')
        self.ln()

    def chapter_text(self, title):
        self.set_font('Arial', '', 12)
        self.cell(30, 10, title)
        self.ln()

def make_book(mazedir):
    pdf = PDF()
    mazefiles = glob(f'{mazedir}/*.png')
    data = [ f.split('-') + [f] for f in mazefiles ]
    data = [
        {
            'type': d[0].split('/')[-1],
            'size': d[2],
            'id': int(d[3]),
            'mode': d[-2].split('.')[0],
            'file': d[-1],
        } for d in data
    ]
    df = pd.DataFrame(data)
    print(df)

    for nmaze, (tags, dg) in enumerate(df.groupby(['type', 'size', 'id'])):
        print(tags)

        if len(dg) != 3:
            print(f'Error with tag : {tags}')
            continue

        image_w = 0.9 * pdf.w
        image_h = image_w
        offset_w = (pdf.w - image_w) / 2

        image_h1 = 0.4 * pdf.h
        image_w1 = image_h1
        offset_w1 = (pdf.w - image_w1) / 2

        mazeimg = dg[ dg['mode'] == 'labyrinth' ].file.values[0]
        dbgimg = dg[ dg['mode'] == 'debug' ].file.values[0]
        solimg = dg[ dg['mode'] == 'solved' ].file.values[0]
        print(mazeimg)

        pdf.add_page()
        pdf.chapter_title(f'Maze #{nmaze+1}')
        pdf.chapter_text(f'Type {tags[0]} size {tags[1]} id {nmaze}')
        pdf.image(mazeimg, offset_w, 50, image_w, image_h)

        pdf.add_page()
        pdf.chapter_title(f'Maze #{nmaze+1} - Solution and Debug')
        pdf.image(solimg, offset_w1, 40, image_w1, image_h1)
        pdf.image(dbgimg, offset_w1, 40 + image_h1, image_w1, image_h1)

    pdf.output(f'{mazedir}/mazebook.pdf', 'F')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--book', action='store_true')
    args = parser.parse_args()

    if args.book:
        mazedir = 'mazes/book'
        make_book(mazedir)
