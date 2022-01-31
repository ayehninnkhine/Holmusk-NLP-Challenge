import gensim
import pandas as pd


def vocab_count(word2vec):
    return len(word2vec.index_to_key)


def get_w2v_analysis(w2v, medical_concept, threshold):
    not_found_count_term1 = 0
    not_found_count_term2 = 0
    not_found_list_term1 = []
    not_found_list_term2 = []
    cosine_count = 0
    for index, row in medical_concept.iterrows():
        term1 = row["Term1"].lower()
        term2 = row["Term2"].lower()
        if term1 not in w2v.index_to_key and term1 not in not_found_list_term1:
            not_found_list_term1.append(term1)
            not_found_count_term1 += 1
        if term2 not in w2v.index_to_key and term2 not in not_found_list_term2:
            not_found_list_term2.append(term2)
            not_found_count_term2 += 1
        if term1 in w2v.index_to_key and term2 in w2v.index_to_key:
            cosine_score = w2v.similarity(term1, term2)
            if cosine_score >= threshold:
                cosine_count += 1
    return not_found_count_term1, not_found_count_term2, cosine_count


word2vec_path = r"C:\Users\61102\PSU-PhD\Holmusk\BioWordVec_PubMed_MIMICIII_d200.vec.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, no_header=False)

vocab = vocab_count(word2vec)
print("Number of Vocabulary=", vocab)

medical = pd.read_csv(r"C:\Users\61102\PSU-PhD\Holmusk\MedicalConcepts.csv")
term1_not_found, term_2_not_found, cosine_count = get_w2v_analysis(word2vec, medical, 0.6)
print(term1_not_found)
print(term_2_not_found)
print(cosine_count)
