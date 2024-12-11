import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Configuración inicial
st.title("Chatbot basado en RAG")
st.write("Este chatbot utiliza Retrieval-Augmented Generation para responder preguntas basadas en el CV del alumno.")

# Cargar variables de entorno
load_dotenv()

# Verificar las claves API
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Mostrar estado de las claves
st.write("Estado de las claves API:")
st.write("- PINECONE_API_KEY:", "Configurada ✅" if pinecone_api_key else "No encontrada ❌")
st.write("- GROQ_API_KEY:", "Configurada ✅" if groq_api_key else "No encontrada ❌")

if not pinecone_api_key or not groq_api_key:
    st.error("Por favor, verifica que las claves API estén configuradas en el archivo .env")
    st.stop()

# Inicializar modelos
try:
    # Inicializar Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("multilingual-e5-large")
    
    # Inicializar el modelo de embeddings
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    # Inicializar cliente de Groq
    client = Groq(api_key=groq_api_key)
    CHAT_MODEL = "llama-3.1-70b-versatile"
    
    st.success("Conexión exitosa con todos los servicios")
except Exception as e:
    st.error(f"Error al inicializar servicios: {str(e)}")
    st.stop()

# Función para obtener embeddings
def get_embedding(text):
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error al generar embeddings: {str(e)}")
        return None

# Función para buscar información similar
def buscar_informacion_similar(query, top_k=3):
    query_embedding = get_embedding(query)
    if query_embedding:
        try:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_values=True,
                include_metadata=True
            )
            return results
        except Exception as e:
            st.error(f"Error al buscar información similar: {str(e)}")
    return None

# Subir CV
uploaded_file = st.file_uploader("Sube tu CV en formato de texto", type=["txt"])

if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode("utf-8")
        st.write("Contenido extraído del CV:")
        st.write(content)

        # Generar embeddings y subir a Pinecone
        embedding = get_embedding(content)
        if embedding:
            try:
                index.upsert(
                    vectors=[{
                        "id": "cv_content",
                        "values": embedding,
                        "metadata": {"texto": content}
                    }]
                )
                st.success("CV procesado y almacenado correctamente")
            except Exception as e:
                st.error(f"Error al almacenar en Pinecone: {str(e)}")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")

# Área para preguntas
st.write("---")
st.write("## Hacer preguntas sobre el CV")
query = st.text_input("Escribe tu pregunta aquí:")

if query:
    try:
        resultados = buscar_informacion_similar(query)
        if resultados and resultados.matches:
            contexto = resultados.matches[0].metadata.get("texto", "")
            prompt = f"""
            Basándote en el siguiente contexto del CV:
            {contexto}
            
            Responde a la siguiente pregunta:
            {query}
            """
            
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": "Eres un asistente especializado en analizar CVs y responder preguntas sobre ellos."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                st.write("### Respuesta:")
                st.write(response.choices[0].message.content)
            except Exception as e:
                if "rate limit exceeded" in str(e).lower():
                    st.error("Se ha excedido el límite de uso. Por favor, espera un momento antes de intentar nuevamente.")
                else:
                    st.error(f"Error al generar la respuesta: {str(e)}")
        else:
            st.warning("No se encontró información relevante en el CV para responder esta pregunta.")
            
    except Exception as e:
        st.error(f"Error al procesar la pregunta: {str(e)}")