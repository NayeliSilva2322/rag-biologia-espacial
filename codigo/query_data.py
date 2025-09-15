import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

CHROMA_PATH = "chroma"  

PROMPT_TEMPLATE = """
Responde la pregunta basándote únicamente en el siguiente contexto:

{context}

---

Responde la pregunta basándote en el contexto anterior: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Verificar que existe la base de datos
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: La base de datos {CHROMA_PATH} no existe.")
        print("Primero ejecuta populate_database.py para crear la base de datos.")
        return

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0:
        print(f"No se encontraron documentos para: '{query_text}'")
        return
    
    # Normalizar scores de cosine similarity (-1 to 1) a (0 to 1)
    normalized_results = []
    for doc, score in results:
        normalized_score = (score + 1) / 2  # De [-1,1] a [0,1]
        normalized_results.append((doc, normalized_score))
    
    # Ordenar por relevancia (score más alto = más relevante)
    normalized_results.sort(key=lambda x: x[1], reverse=True)
    print(f"Relevancia máxima encontrada: {normalized_results[0][1]:.3f}")
    # Usar umbral más realista
    if results[0][1] < -10:  # Umbral más bajo
        print(f"No se encontraron resultados suficientemente relevantes para: '{query_text}'")
        print(f"Relevancia máxima encontrada: {normalized_results[0][1]:.3f}")
        return

    print(f"Encontrados {len(results)} documentos relevantes")
    

    # Preparar contexto
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY no encontrada en variables de entorno.")
        print("Agrega tu API key al archivo .env")
        return
    
    # Crear el modelo LLM
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")

    # Crear la cadena LCEL
    rag_chain = (
        {
            "context": lambda x: x["context"], 
            "question": lambda x: x["question"]
        }
        | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        | llm
        | StrOutputParser()
    )
    
    try:
        # Invocar la cadena
        response_text = rag_chain.invoke({
            "context": context_text, 
            "question": query_text
        })

        # Preparar fuentes
        sources = [doc.metadata.get("source", "Fuente desconocida") for doc, _score in normalized_results[:3]]
        
        # Mostrar respuesta formateada
        print("\n" + "="*50)
        print("RESPUESTA:")
        print("="*50)
        print(response_text)
        print("\n" + "="*50)
        print("FUENTES:")
        print("="*50)
        for i, source in enumerate(sources, 1):
            print(f"{i}. {os.path.basename(source) if source != 'Fuente desconocida' else source}")
        
    except Exception as e:
        print(f"Error al generar respuesta: {str(e)}")

if __name__ == "__main__":
    main()