import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image



image = Image.open('dna.jpg')
st.image(image, use_column_width=True)

st.write("""
#DNA  Nucleotide Count web App
         
This app counts the nucleotide composition of query DNNA!
         
""")

st.header('Enter DNA sequence')

sequence_input = ">DNA Query\nGGGCCCCCCCCCATTTGGCCCAATATC\nTGTGGGGGGGGGGGCCCC\nCCCAAAAAAATTTGCGCG\nCTATATATCTTCTCCTTAATGGG"


sequence = st.textarea("Sequence_Input", sequence_input, height=250)
sequence = sequence.splitlines()
sequence = sequence[1:]
sequence = ''.join(sequence)

st.write("""
***
""")

st.header('INPUT (DNA Query)')
sequence


#dictionary
st.subheader('1.Print Dictionary')
def DNA_nucleotide_count(seq):
    d = dict([
        ('A',seq.count('A')),
        ('T',seq.count('T')),
        ('G',seq.count('G')),
        ('C',seq.count('C'))
        ])
    return d

X = DNA_nucleotide_count(sequence)


#dataframe
st.subheader('3.Display DataFrame')
df = pd.DataFrame.from_dict(X, orient='index')
df = df.rename({0:'count'}, axis='columns')
df.reset_index(inplace=True)
df = df.rename(columns = {'index':'nucleotide'})
st.write(df)


#barchart

st.subheader('4.Display Bar Chart')
p = alt.Chart(df).mark_bar().encode(
    x='nucleotide',
    y='count'
)

st.write(p)

