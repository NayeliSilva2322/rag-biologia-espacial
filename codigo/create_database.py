from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma  
from dotenv import load_dotenv
import os
import shutil
import glob

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"  
DATA_PATH = r"C:\Users\PC\Downloads\rag-1\rag-1-CNSA\datos"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No se pudieron cargar documentos. Terminando...")
        return
    
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    """Cargar documentos PDF de forma más robusta"""
    documents = []
    
    # Verificar que la carpeta existe
    if not os.path.exists(DATA_PATH):
        print(f"Error: La carpeta {DATA_PATH} no existe.")
        return []
    
    # Buscar archivos PDF
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"No se encontraron archivos PDF en {DATA_PATH}")
        return []
    
    print(f"Encontrados {len(pdf_files)} archivos PDF")
    
    # Cargar cada PDF individualmente
    for pdf_file in pdf_files:
        try:
            print(f"Cargando: {os.path.basename(pdf_file)}")
            loader = PyPDFLoader(pdf_file)
            doc_pages = loader.load()
            documents.extend(doc_pages)
            print(f"Cargado exitosamente ({len(doc_pages)} páginas)")
            
        except Exception as e:
            print(f"Error cargando {os.path.basename(pdf_file)}: {str(e)}")
            continue
    
    print(f"\nTotal documentos cargados: {len(documents)}")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Mostrar ejemplo solo si hay chunks
    if len(chunks) > 10:
        document = chunks[10]
        print("Ejemplo de chunk:")
        print(document.page_content)
        print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    #db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()