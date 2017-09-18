from bs4 import BeautifulSoup  # utilizado para extrair o texto de um html
import time
import re

def extract_word_set(path):
    """
    Extracts set of unique words on document
    :return: set of words(string)
    """

    start_time = time.time()  # estimativa do tempo para processamento do database

    remove = "!:.@()-{}/"  # caracteres especiais para serem removidos

    html_file = open(path, 'r',encoding='utf-8', errors='ignore'    )  # abre o arquivo específico
    soup = BeautifulSoup(html_file, "html.parser")
    html_content = soup.get_text()  # extrai o texto do arquivo .html

    spaces = " " * remove.__len__()  # utilizado para remover os caracteres
    html_content = html_content.translate(str.maketrans(remove, spaces))  # remove os caracteres desejados
    html_line = " ".join(html_content.splitlines()[6:])  # remove um cabeçalho que tem nos arquivos antes do <html>
    words = html_line.split()  # separa o string em palavras

    unique_words = set(words)  # cria um set de palavras únicas


    print(sorted(unique_words)) # imprime as palavras em ordem alfabética


    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds --- (3000)" % (3000*(time.time() - start_time)))

    return unique_words


def extract_text(path):
    """
    Extracts set of unique words on document
    :return: set of words(string)
    """

    start_time = time.time()  # estimativa do tempo para processamento do database

    remove = "!:.@()-{}/"  # caracteres especiais para serem removidos

    html_file = open(path, 'r',encoding='utf-8', errors='ignore')  # abre o arquivo específico
    soup = BeautifulSoup(html_file, "html.parser")
    html_content = soup.get_text()  # extrai o texto do arquivo .html

    spaces = " " * remove.__len__()  # utilizado para remover os caracteres
    html_content = html_content.translate(str.maketrans(remove, spaces))  # remove os caracteres desejados
    html_line = " ".join(html_content.splitlines()[6:])  # remove um cabeçalho que tem nos arquivos antes do <html>
    html_line = re.sub("\d+", "", html_line)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("--- %s seconds --- (3000)" % (3000*(time.time() - start_time)))

    return html_line