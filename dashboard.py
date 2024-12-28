import streamlit as st
import tempfile

from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_groq import ChatGroq

from uploader import *

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'Pdf', 'Csv', 'Texto'
]

CONFIG_MODELOS = {'Groq': {'modelos': ['gemma2-9b-it','llama-3.3-70b-versatile'], 'chat': ChatGroq},
                  'OpenAir': {'modelos': ['gpt-4o-mini','gpt-3.5-turbo-0125'], 'chat': OpenAI},}

MEMORIA = ConversationBufferMemory()

def carrega_arquivo(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documentos = carrega_url(arquivo)
    if tipo_arquivo == 'Youtube':
        documentos = carrega_youtube(arquivo)
    if tipo_arquivo == 'Pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome = temp.name
        documentos = carrega_pdf(nome)
    if tipo_arquivo == 'Csv':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome = temp.name
        documentos = carrega_csv(nome)
    if tipo_arquivo == 'Texto':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome = temp.name
        documentos = carrega_txt(nome)
    return documentos

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    documento = carrega_arquivo(tipo_arquivo, arquivo)

    system_message = '''
        Você é um assistente amigável chamado Oráculo.
        Você possui acesso às seguintes informações vindas 
        de um documento {}: 
    
        ####
        {}
        ####
    
        Utilize as informações fornecidas para basear as suas respostas.
    
        Sempre que houver $ na sua saída, substita por S.
    
        Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
        sugira ao usuário carregar novamente o Oráculo!
    
    '''.format(tipo_arquivo, documento)
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain

def pagina_chat():
    st.header('🤖 Bem-vindo ao chatbot do Oráculo', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error('Carregue o Oráculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)

    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Fale com o Oráculo')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
        }))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

def sidebar():
    tabs = st.tabs(['Upload de arquivos', 'Seleção de modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do site')
        if tipo_arquivo == 'Notion':
            arquivo = st.text_input('Digite a url do notion')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do video')
        if tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faca o uload do arquivo PDF', type=['pdf'])
        if tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faca o uload do arquivo CSV', type=['csv'])
        if tipo_arquivo == 'Texto':
            arquivo = st.file_uploader('Faca o uload do arquivo TXT', type=['txt'])

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Digite a API do modelo {provedor}',
            value=st.session_state.get(f'api_key_{provedor}')
        )

        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar Oráculo', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Limpar históricos', use_container_width=True):
        st.session_state['memoria'] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()