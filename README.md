## Setup

Para utilizar o código deste repositório, siga as instruções a seguir:

Crie um ambiente virtual do Python:

``` shell
python3 -m venv env
```

Ative o ambiente virtual (**você deve fazer isso sempre que for executar algum script deste repositório**):

``` shell
source ./env/bin/activate
```
OU:
``` shell
.\env\Scripts\activate 
```

Instale as dependências com:

``` shell
python3 -m pip install -r requirements.txt --upgrade
```


Rode localmente com:

``` shell
mkdocs serve -o
```

Para subir no Git Hub Pages:

```shell
mkdocs gh-deploy
```

