# LangChain RAG Tutorial

## Instalación de dependencias

1.  Antes de instalar las dependencias del archivo `requirements.txt`,
    es recomendable realizar lo siguiente debido a problemas actuales al
    instalar `onnxruntime` con `pip install onnxruntime`.

    -   Para usuarios de **MacOS**, una solución alternativa es instalar
        primero la dependencia `onnxruntime` para `chromadb` usando:

    ``` bash
    conda install onnxruntime -c conda-forge
    ```

2.  Ahora ejecuta este comando para instalar las dependencias del
    archivo `requirements.txt`:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Instala las dependencias necesarias para trabajar con archivos
    Markdown:

    ``` bash
    pip install "unstructured[md]"
    ```

## Crear la base de datos

Crea la base de datos en **Chroma**:

``` bash
python create_database.py
```

## Consultar la base de datos

Realiza consultas a la base de datos en **Chroma**.\
Ejemplo de consulta:

``` bash
python query_data.py "DAME UN RESUMEN DE TUS DATOS?"
```
