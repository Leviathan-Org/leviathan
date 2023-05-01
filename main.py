import calendar
from cmath import cos
import datetime
from datetime import date
from distutils.log import Log
from locale import normalize
import logging
from functools import reduce
from statistics import mode
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import requests
from structlog import get_logger, configure
import structlog
from structlog.stdlib import LoggerFactory
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import sys
import openai

BASE = "davinci:ft-personal-2022-08-31-04-45-44"
MODEL = "davinci:ft-personal-2022-08-30-21-57-22"

MAX_TOKENS = 31 
NUCLEUS = .95

GLOBAL_PROMPT_LIST = ["test", "test1"]

# configure logger
today = date.today().strftime("%m_%d_%Y")
logging.basicConfig(filename='log/history_{}'.format(today), encoding='utf-8', level=logging.DEBUG)
configure(
    processors=[
        structlog.processors.UnicodeEncoder(),
        structlog.processors.KeyValueRenderer(
            key_order=['event', 'model', 'prompt', 'data'],
        ),
    ], 
    logger_factory=LoggerFactory())
log = get_logger()

matplotlib.rcParams.update({'font.size': 4})

# THESE API KEYS NO LONGER VALID
# # configure openai client
# openai.organization = ('org-UQAqGjWYD9fRg5PJebJV7gkt')
# # TODO: store env variables elsewhere
# openai.api_key = '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'

# # configure huggingface
# api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/bert-base-uncased"
# headers = {"Authorization": f"Bearer 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"}

def parse_prompts(filename):
    words = []
    with open(filename, 'r') as f:
        for line in f:
            word = line.strip()
            words.append(word)
    return words

def parse_labels_and_completions(filename):
    labels = []
    completions = []
    with open(filename, 'r') as f:
        for line in f:
            l = line.strip().split(':')
            label = l[0]
            completion = l[1]

            labels.append(label)
            completions.append(completion)
    
    return labels, completions

def generate_and_save_completions(prompts, filename):
    with open(filename, "w+") as f:
        for p in prompts:
            c = completion(BASE, MAX_TOKENS, NUCLEUS, prompt=p).replace("\n", "")
            f.write("{}: {}\n".format(p, c))
    return 

def initialize_vector_field():
    return

def slope(y, x):
    x_endpadded = np.zeros(np.size(x)+2, dtype=x.dtype.char)
    x_endpadded[0]    = x[0]
    x_endpadded[1:-1] = x 
    x_endpadded[-1]   = x[-1]

    y_endpadded = np.zeros(np.size(y)+2, dtype=y.dtype.char)
    y_endpadded[0]    = y[0]
    y_endpadded[1:-1] = y
    y_endpadded[-1]   = y[-1]

    y_im1 = y_endpadded[:-2]
    y_ip1 = y_endpadded[2:]
    x_im1 = x_endpadded[:-2]
    x_ip1 = x_endpadded[2:]

    return (y_ip1 - y_im1) / (x_ip1 - x_im1)

def curl(x, y, Fx, Fy):
    """Curl of a vector F on a 2-D "rectangular" grid.
    """

    dFy_dx = np.zeros( (len(y), len(x)), typecode=np.Float )
    dFx_dy = np.zeros( (len(y), len(x)), typecode=np.Float )

    for iy in range(len(y)):
        dFy_dx[iy,:] = slope( np.ravel(Fy[iy,:]), x )

    for ix in range(len(x)):
        dFx_dy[:,ix] = slope( np.ravel(Fx[:,ix]), y )

    return dFy_dx - dFx_dy


def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    print("NUM_DIMS: {}".format(num_dims))
    return np.ufunc.reduce(np.add, [np.gradient(f[i]) for i in range(num_dims)])

def decoupled_gradient(f):
    num_dims = len(f)
    return [np.gradient(f[i]) for i in range(num_dims)]

def determinant_ish(f):
    num_dims = len(f)
    return [np.gradient(f[i]) for i in range(num_dims -1 , -1, -1)]    

def list_models():
    return openai.Model.list()

def query_embeddings(inputs):
    response = requests.post(api_url, headers=headers, json={"inputs": inputs, "options":{"wait_for_model":True}})
    return response.json()

def completion(model, tokens, nucleus, prompt):
    response = openai.Completion.create(
        model=model,
        max_tokens=tokens,
        top_p=nucleus,
        temperature=1,
        prompt=prompt,
        echo=True,
    )
    log.info('Completion', model=model, prompt=prompt, data=response)
    return response["choices"][0]["text"]


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-d':
        filename = args[1]
        labels, completions = parse_labels_and_completions(filename)

        embeddings = []
        for c in completions:
            e = query_embeddings([c])
            embeddings.append(e)
        
        df = pandas.DataFrame(embeddings)

        ## Clean the data by reducing tokens into one array extending the arrays to the same length 
        vecs = []
        max_len = 0
        for i in range(len(df.values)):
            aggregate_vec = np.array(reduce(list.__add__, df.values[i][0]))
            max_len = max(max_len, len(aggregate_vec))
            vecs.append(aggregate_vec)
        
        extended_vecs = []
        for v in vecs:
            e = v.copy()
            e.resize(max_len)
            extended_vecs.append(e)
        
        # Want to visualize divergence in 2-d (x,y) and sentence vectors have many dimensions,
        # project the data down to it's two most significant dimensions       
        pca = PCA(n_components=2)
        pca_vecs = pca.fit_transform(extended_vecs)

        ## each sentence now is represented by one x,y pair, get the x and y values for each sentence
        x, y = zip(*pca_vecs)
        
        ## given scattered x, y points, derive an underlying curve and find the partial derivates of the curve at 
        ## each x, y point (rate of change for each direction)
        div = decoupled_gradient([x, y])

        print("DIV: {}".format(div))

        u = []
        for i, deriv in zip(x, div[0]):
            val = float(i)**float(deriv)
            u.append(val)
        
        v = []
        for j, deriv in zip(y, div[1]):
            val = float(j)**float(deriv)
            v.append(val)

        dg_dx, df_dy = determinant_ish([x, y])
        z = np.subtract(dg_dx, df_dy)
        
        
        print("x: {}".format(x))
        print("y: {}".format(y))
        print("curl: {}".format(z))

        d = divergence([z])

        sum = 0.0
        for ele in d:
            sum += ele

        print("del dot curl: {}".format(d))
        print("sum {}".format(sum))
        # print("u: {}".format(u))
        # print("v: {}".format(v))

        date = datetime.datetime.utcnow()
        utc_time = calendar.timegm(date.utctimetuple())
        plt.figure(dpi=1200)
        plt.quiver(x, y, u, v)
        plt.savefig('log/plots/_{}_{}_.png'.format(today, utc_time))
        plt.show() 

        return
        
    if args[0] == '-v':
        d = datetime.datetime.utcnow()
        utc_time = calendar.timegm(d.utctimetuple())
        completion_filename = "log/completions/completions_{}_{}.txt".format(today, utc_time)

        prompts = parse_prompts("completions_10_18_2022_1666127961.txt")

        generate_and_save_completions(prompts, completion_filename)
        # labels, completions = parse_labels_and_completions(completion_filename)

        # embeddings = []
        # for c in completions:
        #     e = query_embeddings([c])
        #     embeddings.append(e)
        
        # print(embeddings)
        # df = pandas.DataFrame(embeddings)

        # ## Clean the data by reducing tokens into one array extending the arrays to the same length 
        # vecs = []
        # max_len = 0
        # for i in range(len(df.values)):
        #     aggregate_vec = np.array(reduce(list.__add__, df.values[i][0]))
        #     max_len = max(max_len, len(aggregate_vec))
        #     vecs.append(aggregate_vec)
        
        # extended_vecs = []
        # for v in vecs:
        #     e = v.copy()
        #     e.resize(max_len)
        #     extended_vecs.append(e)

        # ## Want to visualize divergence in 2-d (x,y) and sentence vectors have many dimensions,
        # # project the data down to it's two most significant dimensions       
        # pca = PCA(n_components=2)
        # pca_vecs = pca.fit_transform(extended_vecs)

        # ## each sentence now is represented by one x,y pair, get the x and y values for each sentence
        # x, y = zip(*pca_vecs)
        
        # ## given scattered x, y points, derive an underlying curve and find the partial derivates of the curve at 
        # ## each x, y point (rate of change for each direction)
        # div = decoupled_gradient([x, y])

        # print("DIV: {}".format(div))

        # u = []
        # for i, deriv in zip(x, div[0]):
        #     val = float(i)**float(deriv)
        #     u.append(val)
        
        # v = []
        # for j, deriv in zip(y, div[1]):
        #     val = float(j)**float(deriv)
        #     v.append(val)
        
        
        # print("x: {}".format(x))
        # print("y: {}".format(y))
        # print("u: {}".format(u))
        # print("v: {}".format(v))
        

        # # ax = plt.figure().add_subplot(projection='3d')
        # # idx = 0
        # # for i, j in zip(x, y):
        # #     label = labels[idx]
        # #     plt.annotate(
        # #         label,
        # #         (x, y),
        # #         textcoords="offset points",
        # #         xytext=(0,10), 
        # #         ha='center'
        # #     )
        # #     idx+=1
        
        # date = datetime.datetime.utcnow()
        # utc_time = calendar.timegm(date.utctimetuple())
        # plt.figure(dpi=1200)
        # plt.quiver(x, y, u, v)
        # plt.savefig('log/plots/_{}_{}_.png'.format(today, utc_time))
        # plt.show()
        return    
            
    if len(args) == 2 and args[0] == "-p":
        # Genereate completion for both models
        # prompt = args[1]
        # base_completion = completion(BASE, MAX_TOKENS, NUCLEUS, prompt=prompt)
        # model_completion = completion(MODEL, MAX_TOKENS, NUCLEUS, prompt=prompt)

        # # Obtain vectors for each output and calculate cosine distance
        # output = query_embeddings([base_completion, model_completion]) 
        # df = pandas.DataFrame(output)
        # # df.to_csv('embeddings.txt')
        # base_vec = np.array(reduce(list.__add__, df.values[0][0]))
        # model_vec = np.array(reduce(list.__add__, df.values[1][0]))
        # # base_vec_norm = np.linalg.norm(df.values[0][0])
        # # model_vec_norm = np.linalg.norm(df.values[1][0])
        # # print(base_vec_norm)
        # # print(model_vec_norm)

        # # base_vec = np.reshape(df.values[0][0], (1, -1))
        # # model_vec = np.reshape(df.values[1][0], (1, -1))

        # max_len = max(len(base_vec), len(model_vec))
        # b = base_vec.copy()
        # m = model_vec.copy()
        
        # b.resize(max_len)
        # m.resize(max_len)
        # # B = np.append(base_vec, np.zeros(max_len-len(base_vec)))
        # # M = np.append(model_vec, np.zeros(max_len-len(model_vec)))

        # print("MAX LENGTH: {}".format(max_len))


        # print("BASE COMPLETION: [ {} ]".format(base_completion))
        # print("MODEL COMPLETION: [ {} ]".format(model_completion))
        # # print("BASE EMBEDDING:  {} ".format(embeddings[0]))
        # # print("MODEL EMBEDDING: {}".format(embeddings[1]))
        # # print("MODEL EMBEDDING:  {} ".format(embeddings[1]))
        # print("COSINE DISTANCE: {}".format(cosine(b, m)))

        # # x,y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))

        # # u = x/np.sqrt(x**2 + y**2)
        # # v = y/np.sqrt(x**2 + y**2)

        # # plt.quiver(x,y,u,v)
        # # plt.show()

        prompt = args[1]
        c = completion(BASE, MAX_TOKENS, NUCLEUS, prompt=prompt)
        print(c)

    return

if __name__ == "__main__":
    main()
