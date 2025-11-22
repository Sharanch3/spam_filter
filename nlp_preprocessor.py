import spacy

# download the spacy model
nlp = spacy.load("en_core_web_sm")


# helper functions from text preprocessing section
def lower_replace(series):
    
    '''
    This function applies:
    1. lowercasing the text
    2. remove puncutations
    3. remove[...] whatever inside the bracket
    4. remove links
     
    '''
    output = series.str.lower()
    output = output.str.replace(r'\[.*?\]', '', regex=True)
    output = output.str.replace(r'[^\w\s]', '', regex=True)   
    output = output.str.replace(r'https?://\S+|www\.\S+', '', regex=True) 
    return output



def token_lemma_nonstop(text):
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(output)




def nlp_pipeline(series):
    output = lower_replace(series)
    output = output.apply(token_lemma_nonstop)
    return output


