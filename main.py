import re
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PDF_FILE = 'PDF/Русский язык 5 класс.pdf'

# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


# Функция для поиска релевантных фрагментов текста
def search_text(query, text, top_n=3):
    sentences = text.split('. ')
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform([query] + sentences)
    cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    return [(sentences[i], cosine_similarities[i]) for i in top_indices]


def main():
    # Путь к вашему PDF-документу
    pdf_path = PDF_FILE
    text = extract_text_from_pdf(pdf_path)

    query = input("Введите ваш поисковый запрос: ")

    relevant_fragments = search_text(query, text)

    print("\nТоп-3 релевантных фрагмента:")
    for fragment, score in relevant_fragments:
        fragment = re.sub('\n', '', fragment.strip())
        print(f"- {fragment} (Скор: {score:.4f})")


if __name__ == "__main__":
    main()
