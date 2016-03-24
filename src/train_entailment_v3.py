import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time

from logistic_sgd import LogisticRegression
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from Thang import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from loadData import load_MCTest_corpus_DPNQ, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Conv_with_input_para_one_col_featuremap, Average_Pooling_for_Top, create_conv_para, Average_Pooling, create_highw_para, Average_Pooling_Scan
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import linalg, mat, dot
'''
from preprocess_wikiQA import compute_map_mrr

need to change:
1) add dev into training set
2) pretrain the model over an extra dataset
3) question classification

'''

def store_model_to_file(best_params, best_epoch, best_acc):
    # this will overwrite current contents
    fname = '../data/MCTest/Best_Para_at'+str(best_epoch)+'_'+str(best_acc)
    with open(fname, 'wb') as save_file:
        for para in best_params:           
            cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL

def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi#.reshape((1,1))    

def compute_corr(test_y, test_prop):
    if len(test_y)%3!=0 or len(test_prop)%3!=0:
        print 'len(test_y)%3!=0 or len(test_prop)%3!=0'
        print len(test_y), len(test_prop)
        exit(0)
    size=len(test_y)
    batch=3
    n_batches=size/batch
    
    batch_start=list(numpy.arange(n_batches)*batch)
    corr=0
    for start in batch_start:
        sub_y=test_y[start:start+batch]
        sub_prop=test_prop[start:start+batch]
        succ=True
        for i in range(batch):
            if sub_y[i]<sub_prop[i]:
                succ=False
        if succ is True:
            corr+=1
    return corr
        

def evaluate_lenet5(
        learning_rate=0.001,
        n_epochs=2000,
        nkerns=[90,90],
        batch_size=1,
        window_width=2,
        maxSentLength=64,
        maxDocLength=60,
        emb_size=50,
        hidden_size=200,
        L2_weight=0.0065,
        update_freq=1,
        norm_threshold=5.0,
        max_s_length=57,
        max_d_length=59,
        margin=0.2
):
    maxSentLength = max_s_length+2*(window_width-1)
    maxDocLength = max_d_length+2*(window_width-1)
    model_options = locals().copy()
    print "model options", model_options
    rootPath='../data/MCTest/'
    rng = numpy.random.RandomState(23455)
    train_data, train_size, test_data, test_size, vocab_size=load_MCTest_corpus_DPNQ(
        rootPath+'vocab_DQAAAA.txt', # DPNQ.txt',
        rootPath+'mc500.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt_DPN.txt_DPNQ.txt',
        rootPath+'mc500.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt_DPN.txt_DPNQ.txt',
        max_s_length,
        maxSentLength,
        maxDocLength
    ) # vocab_size contains train, dev and test
    [train_data_D,
     train_data_A1,
     train_data_A2,
     train_data_A3,

     train_Label,
     train_Length_D,train_Length_D_s,
     train_Length_A1,
     train_Length_A2,
     train_Length_A3,
     train_leftPad_D,train_leftPad_D_s,
     train_leftPad_A1,
     train_leftPad_A2,
     train_leftPad_A3,
     train_rightPad_D,train_rightPad_D_s,
     train_rightPad_A1,
     train_rightPad_A2,
     train_rightPad_A3
    ] = train_data
    [test_data_D,
     test_data_A1,
     test_data_A2,
     test_data_A3,
     test_Label,
     
     test_Length_D,test_Length_D_s,
     test_Length_A1,
     test_Length_A2,
     test_Length_A3,
     test_leftPad_D,test_leftPad_D_s,
     test_leftPad_A1,
     test_leftPad_A2,
     test_leftPad_A3,
     test_rightPad_D,test_rightPad_D_s,
     test_rightPad_A1,
     test_rightPad_A2,
     test_rightPad_A3
    ] = test_data                

    n_train_batches = train_size/batch_size
    n_test_batches = test_size/batch_size
    
    train_batch_start = list(numpy.arange(n_train_batches)*batch_size)
    test_batch_start = list(numpy.arange(n_test_batches)*batch_size)

        rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)

    rand_values=load_word2vec_to_init(rand_values, rootPath+'vocab_DPNQ_glove_50d.txt')

    embeddings=theano.shared(value=rand_values, borrow=True)      
    
    error_sum=0
    
    # allocate symbolic variables for the data
    index = T.lscalar()
    index_D = T.lmatrix()   # now, x is the index matrix, must be integer

    index_A1= T.lvector()
    index_A2= T.lvector()
    index_A3= T.lvector()
    
    len_D=T.lscalar()
    len_D_s=T.lvector()

    len_A1=T.lscalar()
    len_A2=T.lscalar()
    len_A3=T.lscalar()


    left_D=T.lscalar()
    left_D_s=T.lvector()

    left_A1=T.lscalar()
    left_A2=T.lscalar()
    left_A3=T.lscalar()


    right_D=T.lscalar()
    right_D_s=T.lvector()

    right_A1=T.lscalar()
    right_A2=T.lscalar()
    right_A3=T.lscalar()

    ishape = (emb_size, maxSentLength)  # sentence shape
    dshape = (nkerns[0], maxDocLength) # doc shape
    filter_words=(emb_size,window_width)
    filter_sents=(nkerns[0], window_width)

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    #layer0_input = x.reshape(((batch_size*4), 1, ishape[0], ishape[1]))
    layer0_D_input = embeddings[index_D.flatten()].reshape((maxDocLength,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    layer0_A1_input = embeddings[index_A1.flatten()].reshape((batch_size,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    layer0_A2_input = embeddings[index_A2.flatten()].reshape((batch_size,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    layer0_A3_input = embeddings[index_A3.flatten()].reshape((batch_size,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    
        
    conv_W, conv_b=create_conv_para(rng, filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]))
    layer0_para=[conv_W, conv_b] 
    conv2_W, conv2_b=create_conv_para(rng, filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]))
    layer2_para=[conv2_W, conv2_b]
    high_W, high_b=create_highw_para(rng, nkerns[0], nkerns[1]) # this part decides nkern[0] and nkern[1] must be in the same dimension
    highW_para=[high_W, high_b]
    params = layer2_para+layer0_para+highW_para#+[embeddings]

    layer0_D = Conv_with_input_para(rng, input=layer0_D_input,
            image_shape=(maxDocLength, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    layer0_A1 = Conv_with_input_para(rng, input=layer0_A1_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    layer0_A2 = Conv_with_input_para(rng, input=layer0_A2_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    layer0_A3 = Conv_with_input_para(rng, input=layer0_A3_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    
    layer0_D_output=debug_print(layer0_D.output, 'layer0_D.output')
    layer0_A1_output=debug_print(layer0_A1.output, 'layer0_A1.output')
    layer0_A2_output=debug_print(layer0_A2.output, 'layer0_A2.output')
    layer0_A3_output=debug_print(layer0_A3.output, 'layer0_A3.output')
       
    layer1_DA1=Average_Pooling_Scan(
        rng,
        input_D=layer0_D_output,
        input_r=layer0_A1_output,
        kern=nkerns[0],
        left_D=left_D,
        right_D=right_D,
        left_D_s=left_D_s,
        right_D_s=right_D_s,
        left_r=left_A1,
        right_r=right_A1, 
        length_D_s=len_D_s+filter_words[1]-1,
        length_r=len_A1+filter_words[1]-1,
        dim=maxSentLength+filter_words[1]-1,
        doc_len=maxDocLength,
        topk=3
    )
    layer1_DA2 = Average_Pooling_Scan(
        rng,
        input_D=layer0_D_output,
        input_r=layer0_A2_output,
        kern=nkerns[0],
        left_D=left_D,
        right_D=right_D,
        left_D_s=left_D_s,
        right_D_s=right_D_s,
        left_r=left_A2,
        right_r=right_A2, 
        length_D_s=len_D_s+filter_words[1]-1,
        length_r=len_A2+filter_words[1]-1,
        dim=maxSentLength+filter_words[1]-1,
        doc_len=maxDocLength,
        topk=3
    )
    layer1_DA3 = Average_Pooling_Scan(
        rng,
        input_D=layer0_D_output,
        input_r=layer0_A3_output,
        kern=nkerns[0],
        left_D=left_D,
        right_D=right_D,
        left_D_s=left_D_s,
        right_D_s=right_D_s,
        left_r=left_A3,
        right_r=right_A3,
        length_D_s=len_D_s+filter_words[1]-1,
        length_r=len_A3+filter_words[1]-1,
        dim=maxSentLength+filter_words[1]-1,
        doc_len=maxDocLength,
        topk=3
    )

    layer2_DA1 = Conv_with_input_para(rng, input=layer1_DA1.output_D.reshape((batch_size, 1, nkerns[0], dshape[1])),
            image_shape=(batch_size, 1, nkerns[0], dshape[1]),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_DA2 = Conv_with_input_para(rng, input=layer1_DA2.output_D.reshape((batch_size, 1, nkerns[0], dshape[1])),
            image_shape=(batch_size, 1, nkerns[0], dshape[1]),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_DA3 = Conv_with_input_para(rng, input=layer1_DA3.output_D.reshape((batch_size, 1, nkerns[0], dshape[1])),
            image_shape=(batch_size, 1, nkerns[0], dshape[1]),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)

    layer2_A1 = Conv_with_input_para_one_col_featuremap(rng, input=layer1_DA1.output_QA_sent_level_rep.reshape((batch_size, 1, nkerns[0], 1)),
            image_shape=(batch_size, 1, nkerns[0], 1),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_A2 = Conv_with_input_para_one_col_featuremap(rng, input=layer1_DA2.output_QA_sent_level_rep.reshape((batch_size, 1, nkerns[0], 1)),
            image_shape=(batch_size, 1, nkerns[0], 1),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_A3 = Conv_with_input_para_one_col_featuremap(rng, input=layer1_DA3.output_QA_sent_level_rep.reshape((batch_size, 1, nkerns[0], 1)),
            image_shape=(batch_size, 1, nkerns[0], 1),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)

    layer2_A1_output_sent_rep_Dlevel=debug_print(layer2_A1.output_sent_rep_Dlevel, 'layer2_A1.output_sent_rep_Dlevel')
    layer2_A2_output_sent_rep_Dlevel=debug_print(layer2_A2.output_sent_rep_Dlevel, 'layer2_A2.output_sent_rep_Dlevel')
    layer2_A3_output_sent_rep_Dlevel=debug_print(layer2_A3.output_sent_rep_Dlevel, 'layer2_A3.output_sent_rep_Dlevel')

    layer3_DA1=Average_Pooling_for_Top(rng, input_l=layer2_DA1.output, input_r=layer2_A1_output_sent_rep_Dlevel, kern=nkerns[1],
                     left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
                      length_l=len_D+filter_sents[1]-1, length_r=1,
                       dim=maxDocLength+filter_sents[1]-1, topk=3)
    layer3_DA2=Average_Pooling_for_Top(rng, input_l=layer2_DA2.output, input_r=layer2_A2_output_sent_rep_Dlevel, kern=nkerns[1],
                     left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
                      length_l=len_D+filter_sents[1]-1, length_r=1,
                       dim=maxDocLength+filter_sents[1]-1, topk=3)
    layer3_DA3=Average_Pooling_for_Top(rng, input_l=layer2_DA3.output, input_r=layer2_A3_output_sent_rep_Dlevel, kern=nkerns[1],
                     left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
                      length_l=len_D+filter_sents[1]-1, length_r=1,
                       dim=maxDocLength+filter_sents[1]-1, topk=3)
    
    #high-way
    

    transform_gate_DA1=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA1.output_D_sent_level_rep) + high_b), 'transform_gate_DA1')
    transform_gate_DA2=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA2.output_D_sent_level_rep) + high_b), 'transform_gate_DA2')
    transform_gate_DA3=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA3.output_D_sent_level_rep) + high_b), 'transform_gate_DA3')

    transform_gate_A1=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA1.output_QA_sent_level_rep) + high_b), 'transform_gate_A1')
    transform_gate_A2=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA2.output_QA_sent_level_rep) + high_b), 'transform_gate_A2')
        
    overall_D_A1=(1.0-transform_gate_DA1)*layer1_DA1.output_D_sent_level_rep+transform_gate_DA1*layer3_DA1.output_D_doc_level_rep
    overall_D_A2=(1.0-transform_gate_DA2)*layer1_DA2.output_D_sent_level_rep+transform_gate_DA2*layer3_DA2.output_D_doc_level_rep
    overall_D_A3=(1.0-transform_gate_DA3)*layer1_DA3.output_D_sent_level_rep+transform_gate_DA3*layer3_DA3.output_D_doc_level_rep

    overall_A1=(1.0-transform_gate_A1)*layer1_DA1.output_QA_sent_level_rep+transform_gate_A1*layer2_A1.output_sent_rep_Dlevel
    overall_A2=(1.0-transform_gate_A2)*layer1_DA2.output_QA_sent_level_rep+transform_gate_A2*layer2_A2.output_sent_rep_Dlevel
    
    simi_sent_level1=debug_print(cosine(layer1_DA1.output_D_sent_level_rep, layer1_DA1.output_QA_sent_level_rep), 'simi_sent_level1')
    simi_sent_level2=debug_print(cosine(layer1_DA2.output_D_sent_level_rep, layer1_DA2.output_QA_sent_level_rep), 'simi_sent_level2')
  
    simi_doc_level1=debug_print(cosine(layer3_DA1.output_D_doc_level_rep, layer2_A1.output_sent_rep_Dlevel), 'simi_doc_level1')
    simi_doc_level2=debug_print(cosine(layer3_DA2.output_D_doc_level_rep, layer2_A2.output_sent_rep_Dlevel), 'simi_doc_level2')
    
    simi_overall_level1=debug_print(cosine(overall_D_A1, overall_A1), 'simi_overall_level1')
    simi_overall_level2=debug_print(cosine(overall_D_A2, overall_A2), 'simi_overall_level2')
 
    simi_1=(simi_overall_level1+simi_sent_level1+simi_doc_level1)/3.0
    simi_2=(simi_overall_level2+simi_sent_level2+simi_doc_level2)/3.0

    simi_PQ=cosine(layer1_DA1.output_QA_sent_level_rep, layer1_DA3.output_D_sent_level_rep)
    simi_NQ=cosine(layer1_DA2.output_QA_sent_level_rep, layer1_DA3.output_D_sent_level_rep)

    #bad matching at overall level

    match_cost=T.maximum(0.0, margin+simi_NQ-simi_PQ) 
    cost=T.maximum(0.0, margin+simi_sent_level2-simi_sent_level1)+T.maximum(0.0, margin+simi_doc_level2-simi_doc_level1)+T.maximum(0.0, margin+simi_overall_level2-simi_overall_level1)
    cost=cost#+match_cost
    
    L2_reg =debug_print((high_W**2).sum()+3*(conv2_W**2).sum()+(conv_W**2).sum(), 'L2_reg')
    
    cost=debug_print(cost+L2_weight*L2_reg, 'cost')
    
    
    test_model = theano.function(
        [index],
        [
            cost,
            simi_sent_level1,
            simi_sent_level2,
            simi_doc_level1,
            simi_doc_level2,
            simi_overall_level1,
            simi_overall_level2
        ],
        givens={
            index_D: test_data_D[index], #a matrix

            index_A1: test_data_A1[index],
            index_A2: test_data_A2[index],
            index_A3: test_data_A3[index],

            len_D: test_Length_D[index],
            len_D_s: test_Length_D_s[index],

            len_A1: test_Length_A1[index],
            len_A2: test_Length_A2[index],
            len_A3: test_Length_A3[index],

            left_D: test_leftPad_D[index],
            left_D_s: test_leftPad_D_s[index],

            left_A1: test_leftPad_A1[index],
            left_A2: test_leftPad_A2[index],
            left_A3: test_leftPad_A3[index],
            
            right_D: test_rightPad_D[index],
            right_D_s: test_rightPad_D_s[index],

            right_A1: test_rightPad_A1[index],
            right_A2: test_rightPad_A2[index],
            right_A3: test_rightPad_A3[index]
        },
        on_unused_input='ignore'
    )
    
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
        updates.append((acc_i, acc))    
  
    train_model = theano.function(
        [index],
        [
            cost, simi_sent_level1, simi_sent_level2, simi_doc_level1, simi_doc_level2, simi_overall_level1, simi_overall_level2
        ],
        updates=updates],
    givens={
        index_D: train_data_D[index],

        index_A1: train_data_A1[index],
        index_A2: train_data_A2[index],
        index_A3: train_data_A3[index],

        len_D: train_Length_D[index],
        len_D_s: train_Length_D_s[index],

        len_A1: train_Length_A1[index],
        len_A2: train_Length_A2[index],
        len_A3: train_Length_A3[index],


        left_D: train_leftPad_D[index],
        left_D_s: train_leftPad_D_s[index],

        left_A1: train_leftPad_A1[index],
        left_A2: train_leftPad_A2[index],
        left_A3: train_leftPad_A3[index],

        
        right_D: train_rightPad_D[index],
        right_D_s: train_rightPad_D_s[index],

        right_A1: train_rightPad_A1[index],
        right_A2: train_rightPad_A2[index],
        right_A3: train_rightPad_A3[index]
    },
    on_unused_input='ignore'
    )

    train_model_predict = theano.function(
        [index],
        [
            cost, simi_sent_level1, simi_sent_level2, simi_doc_level1, simi_doc_level2, simi_overall_level1, simi_overall_level2
        ],
        givens={
            index_D: train_data_D[index],

            index_A1: train_data_A1[index],
            index_A2: train_data_A2[index],
            index_A3: train_data_A3[index],

            len_D: train_Length_D[index],
            len_D_s: train_Length_D_s[index],

            len_A1: train_Length_A1[index],
            len_A2: train_Length_A2[index],
            len_A3: train_Length_A3[index],

            left_D: train_leftPad_D[index],
            left_D_s: train_leftPad_D_s[index],

            left_A1: train_leftPad_A1[index],
            left_A2: train_leftPad_A2[index],
            left_A3: train_leftPad_A3[index],
            
            right_D: train_rightPad_D[index],
            right_D_s: train_rightPad_D_s[index],

            right_A1: train_rightPad_A1[index],
            right_A2: train_rightPad_A2[index],
            right_A3: train_rightPad_A3[index]
        },
        on_unused_input='ignore'
    )



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    mid_time = start_time

    epoch = 0
    done_looping = False
    
    max_acc=0.0
    best_epoch=0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        shuffle(train_batch_start)#shuffle training data


        posi_train_sent=[]
        nega_train_sent=[]
        posi_train_doc=[]
        nega_train_doc=[]
        posi_train_overall=[]
        nega_train_overall=[]
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1
            sys.stdout.write( "Training :[%6f] %% complete!\r" % ((iter%train_size)*100.0/train_size) )
            sys.stdout.flush()
            minibatch_index=minibatch_index+1
            
            cost_average, simi_sent_level1, simi_sent_level2, simi_doc_level1, simi_doc_level2, simi_overall_level1, simi_overall_level2= train_model(batch_start)
            posi_train_sent.append(simi_sent_level1)
            nega_train_sent.append(simi_sent_level2)
            posi_train_doc.append(simi_doc_level1)
            nega_train_doc.append(simi_doc_level2)
            posi_train_overall.append(simi_overall_level1)
            nega_train_overall.append(simi_overall_level2)
            if iter % n_train_batches == 0:
                corr_train_sent=compute_corr(posi_train_sent, nega_train_sent)
                corr_train_doc=compute_corr(posi_train_doc, nega_train_doc)
                corr_train_overall=compute_corr(posi_train_overall, nega_train_overall)
                print 'training @ iter = '+str(iter)+' average cost: '+str(cost_average)+'corr rate:'+str(corr_train_sent*300.0/train_size)+' '+str(corr_train_doc*300.0/train_size)+' '+str(corr_train_overall*300.0/train_size)

            
            if iter % validation_frequency == 0:
                posi_test_sent=[]
                nega_test_sent=[]
                posi_test_doc=[]
                nega_test_doc=[]
                posi_test_overall=[]
                nega_test_overall=[]
                for i in test_batch_start:
                    cost, simi_sent_level1, simi_sent_level2, simi_doc_level1, simi_doc_level2, simi_overall_level1, simi_overall_level2=test_model(i)
                    posi_test_sent.append(simi_sent_level1)
                    nega_test_sent.append(simi_sent_level2)
                    posi_test_doc.append(simi_doc_level1)
                    nega_test_doc.append(simi_doc_level2)
                    posi_test_overall.append(simi_overall_level1)
                    nega_test_overall.append(simi_overall_level2)
                corr_test_sent=compute_corr(posi_test_sent, nega_test_sent)
                corr_test_doc=compute_corr(posi_test_doc, nega_test_doc)
                corr_test_overall=compute_corr(posi_test_overall, nega_test_overall)

                #write_file.close()
                #test_score = numpy.mean(test_losses)
                test_acc_sent=corr_test_sent*1.0/(test_size/3.0)
                test_acc_doc=corr_test_doc*1.0/(test_size/3.0)
                test_acc_overall=corr_test_overall*1.0/(test_size/3.0)
                #test_acc=1-test_score
#                 print(('\t\t\tepoch %i, minibatch %i/%i, test acc of best '
#                            'model %f %%') %
#                           (epoch, minibatch_index, n_train_batches,test_acc * 100.))
                print '\t\t\tepoch', epoch, ', minibatch', minibatch_index, '/', n_train_batches, 'test acc of best model', test_acc_sent*100,test_acc_doc*100,test_acc_overall*100 
                #now, see the results of LR
                #write_feature=open(rootPath+'feature_check.txt', 'w')
                 

  
                find_better=False
                if test_acc_sent > max_acc:
                    max_acc=test_acc_sent
                    best_epoch=epoch    
                    find_better=True     
                if test_acc_doc > max_acc:
                    max_acc=test_acc_doc
                    best_epoch=epoch    
                    find_better=True 
                if test_acc_overall > max_acc:
                    max_acc=test_acc_overall
                    best_epoch=epoch    
                    find_better=True         
                print '\t\t\tmax:',    max_acc,'(at',best_epoch,')'
                if find_better==True:
                    store_model_to_file(params, best_epoch, max_acc)
                    print 'Finished storing best params'  

            if patience <= iter:
                done_looping = True
                break
        
        
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock()
        #writefile.close()
   
        #print 'Batch_size: ', update_freq
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    evaluate_lenet5()
