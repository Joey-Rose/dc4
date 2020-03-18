import spacy
nlp = spacy.load('en_core_web_sm')

piano_text = 'Do Mexican girls like black guys?'.lower()
piano_doc = nlp(piano_text)
for token in piano_doc:
    print (token.text, token.tag_, token.head.text, token.dep_)

print()

piano_text = 'Do girls like black guys with no beard?'.lower()
piano_doc = nlp(piano_text)
for token in piano_doc:
    print (token.text, token.tag_, token.head.text, token.dep_)


# import stanfordnlp
# stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
# nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
# doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
# doc.sentences[0].print_dependencies()