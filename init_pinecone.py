from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

print("Iniciando script de creación del índice...")

# Cargar variables de entorno
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Inicializar Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Verificar índices existentes
print("\nÍndices existentes:")
print(pc.list_indexes().names())

# Intentar crear el índice con la misma configuración que multilingual-e5-large
try:
    pc.create_index(
        name="cv-index",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Usando la misma región que tu índice existente
        )
    )
    print("\n✅ Índice creado exitosamente")
except Exception as e:
    print(f"\nError o índice ya existe: {str(e)}")
    
    # Mostrar información sobre los índices existentes
    try:
        indexes = pc.list_indexes()
        print("\nDetalles de índices existentes:")
        for index in indexes:
            print(f"- {index.name}")
            print(f"  Dimensión: {index.dimension}")
            print(f"  Métrica: {index.metric}")
            print(f"  Especificación: {index.spec}")
    except Exception as e2:
        print(f"No se pudieron obtener detalles de los índices: {str(e2)}")