ر
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58�
j
ConstConst*&
_output_shapes
:*
dtype0*%
valueB*/�<
�
SGDW/output_dense/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*0
shared_name!SGDW/output_dense/bias/momentum
�
3SGDW/output_dense/bias/momentum/Read/ReadVariableOpReadVariableOpSGDW/output_dense/bias/momentum*
_output_shapes
:1*
dtype0
�
!SGDW/output_dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�1*2
shared_name#!SGDW/output_dense/kernel/momentum
�
5SGDW/output_dense/kernel/momentum/Read/ReadVariableOpReadVariableOp!SGDW/output_dense/kernel/momentum*
_output_shapes
:	�1*
dtype0
�
SGDW/dense_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameSGDW/dense_1/bias/momentum
�
.SGDW/dense_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGDW/dense_1/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGDW/dense_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameSGDW/dense_1/kernel/momentum
�
0SGDW/dense_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGDW/dense_1/kernel/momentum* 
_output_shapes
:
��*
dtype0
�
&SGDW/batch_normalization/beta/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&SGDW/batch_normalization/beta/momentum
�
:SGDW/batch_normalization/beta/momentum/Read/ReadVariableOpReadVariableOp&SGDW/batch_normalization/beta/momentum*
_output_shapes	
:�*
dtype0
�
'SGDW/batch_normalization/gamma/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'SGDW/batch_normalization/gamma/momentum
�
;SGDW/batch_normalization/gamma/momentum/Read/ReadVariableOpReadVariableOp'SGDW/batch_normalization/gamma/momentum*
_output_shapes	
:�*
dtype0
�
SGDW/dense/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*+
shared_nameSGDW/dense/kernel/momentum
�
.SGDW/dense/kernel/momentum/Read/ReadVariableOpReadVariableOpSGDW/dense/kernel/momentum*!
_output_shapes
:���*
dtype0
�
SGDW/conv2d_1/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameSGDW/conv2d_1/bias/momentum
�
/SGDW/conv2d_1/bias/momentum/Read/ReadVariableOpReadVariableOpSGDW/conv2d_1/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGDW/conv2d_1/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*.
shared_nameSGDW/conv2d_1/kernel/momentum
�
1SGDW/conv2d_1/kernel/momentum/Read/ReadVariableOpReadVariableOpSGDW/conv2d_1/kernel/momentum*(
_output_shapes
:��*
dtype0
�
SGDW/conv2d/bias/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameSGDW/conv2d/bias/momentum
�
-SGDW/conv2d/bias/momentum/Read/ReadVariableOpReadVariableOpSGDW/conv2d/bias/momentum*
_output_shapes	
:�*
dtype0
�
SGDW/conv2d/kernel/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameSGDW/conv2d/kernel/momentum
�
/SGDW/conv2d/kernel/momentum/Read/ReadVariableOpReadVariableOpSGDW/conv2d/kernel/momentum*'
_output_shapes
:�*
dtype0
{
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�1* 
shared_namefalse_negatives
t
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:	�1*
dtype0
{
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�1* 
shared_namefalse_positives
t
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:	�1*
dtype0
y
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�1*
shared_nametrue_negatives
r
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes
:	�1*
dtype0
y
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�1*
shared_nametrue_positives
r
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:	�1*
dtype0
�
weights_intermediateVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*%
shared_nameweights_intermediate
y
(weights_intermediate/Read/ReadVariableOpReadVariableOpweights_intermediate*
_output_shapes
:1*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:1*
dtype0
z
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:1*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:1*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:1*
dtype0
z
false_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_positives_2
s
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes
:*
dtype0
x
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:*
dtype0
z
false_negatives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_2
s
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes
:*
dtype0
x
true_positives_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_3
q
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
f
	SGDW/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	SGDW/iter
_
SGDW/iter/Read/ReadVariableOpReadVariableOp	SGDW/iter*
_output_shapes
: *
dtype0	
l
weight_decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameweight_decay
e
 weight_decay/Read/ReadVariableOpReadVariableOpweight_decay*
_output_shapes
: *
dtype0
d
momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
momentum
]
momentum/Read/ReadVariableOpReadVariableOpmomentum*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
z
output_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_nameoutput_dense/bias
s
%output_dense/bias/Read/ReadVariableOpReadVariableOpoutput_dense/bias*
_output_shapes
:1*
dtype0
�
output_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�1*$
shared_nameoutput_dense/kernel
|
'output_dense/kernel/Read/ReadVariableOpReadVariableOpoutput_dense/kernel*
_output_shapes
:	�1*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
��*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
w
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*
shared_namedense/kernel
p
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*!
_output_shapes
:���*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:�*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:��*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:�*
dtype0

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:�*
dtype0
�
#serving_default_preprocessing_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_preprocessing_inputConstconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernelbatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/betabatch_normalization/gammadense_1/kerneldense_1/biasoutput_dense/kerneloutput_dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *-
f(R&
$__inference_signature_wrapper_606505

NoOpNoOp
�}
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*�|
value�|B�| B�|
�
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

_init_input_shape* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
�
"layer-0
#layer_with_weights-0
#layer-1
$layer-2
%layer_with_weights-1
%layer-3
&layer-4
'layer-5
(layer_with_weights-2
(layer-6
)layer_with_weights-3
)layer-7
*layer-8
+layer_with_weights-4
+layer-9
,layer_with_weights-5
,layer-10
-layer-11
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
b
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12*
R
40
51
62
73
84
95
:6
=7
>8
?9
@10*
* 
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 

N	capture_0* 
�
	Odecay
Plearning_rate
Qmomentum
Rweight_decay
Siter4momentum�5momentum�6momentum�7momentum�8momentum�9momentum�:momentum�=momentum�>momentum�?momentum�@momentum�*

Tserving_default* 
* 
* 
* 
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ztrace_0* 

[trace_0* 
* 
* 
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

atrace_0* 

btrace_0* 
* 
* 
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

htrace_0* 

itrace_0* 

j_init_input_shape* 
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

4kernel
5bias
 q_jit_compiled_convolution_op*
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses* 
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

6kernel
7bias
 ~_jit_compiled_convolution_op*
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

8kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	9gamma
:beta
;moving_mean
<moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

=kernel
>bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
@bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
b
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12*
R
40
51
62
73
84
95
:6
=7
>8
?9
@10*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEoutput_dense/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEoutput_dense/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*
'
0
1
2
3
4*
4
�0
�1
�2
�3
�4
�5*
* 
* 

N	capture_0* 

N	capture_0* 

N	capture_0* 

N	capture_0* 

N	capture_0* 

N	capture_0* 

N	capture_0* 

N	capture_0* 
* 
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEmomentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEweight_decay1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	SGDW/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*

N	capture_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

N	capture_0* 

N	capture_0* 
* 
* 
* 
* 
* 
* 
* 
* 

40
51*

40
51*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

80*

80*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
 
90
:1
;2
<3*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

?0
@1*

?0
@1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

;0
<1*
Z
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
`
�	variables
�	keras_api
�
thresholds
�true_positives
�false_negatives*
`
�	variables
�	keras_api
�
thresholds
�true_positives
�false_positives*
�
�	variables
�	keras_api
�
init_shape
�true_positives
�false_positives
�false_negatives
�weights_intermediate*
z
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

;0
<1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_3=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_2>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_2>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEweights_intermediateCkeras_api/metrics/4/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�	variables*
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/5/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUESGDW/conv2d/kernel/momentumIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUESGDW/conv2d/bias/momentumIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUESGDW/conv2d_1/kernel/momentumIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUESGDW/conv2d_1/bias/momentumIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUESGDW/dense/kernel/momentumIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE'SGDW/batch_normalization/gamma/momentumIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE&SGDW/batch_normalization/beta/momentumIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUESGDW/dense_1/kernel/momentumIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUESGDW/dense_1/bias/momentumJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUE!SGDW/output_dense/kernel/momentumJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUESGDW/output_dense/bias/momentumJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp'output_dense/kernel/Read/ReadVariableOp%output_dense/bias/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpmomentum/Read/ReadVariableOp weight_decay/Read/ReadVariableOpSGDW/iter/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$true_positives_3/Read/ReadVariableOp%false_negatives_2/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp%false_positives_2/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp(weights_intermediate/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp/SGDW/conv2d/kernel/momentum/Read/ReadVariableOp-SGDW/conv2d/bias/momentum/Read/ReadVariableOp1SGDW/conv2d_1/kernel/momentum/Read/ReadVariableOp/SGDW/conv2d_1/bias/momentum/Read/ReadVariableOp.SGDW/dense/kernel/momentum/Read/ReadVariableOp;SGDW/batch_normalization/gamma/momentum/Read/ReadVariableOp:SGDW/batch_normalization/beta/momentum/Read/ReadVariableOp0SGDW/dense_1/kernel/momentum/Read/ReadVariableOp.SGDW/dense_1/bias/momentum/Read/ReadVariableOp5SGDW/output_dense/kernel/momentum/Read/ReadVariableOp3SGDW/output_dense/bias/momentum/Read/ReadVariableOpConst_1*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *(
f#R!
__inference__traced_save_607432
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense_1/kerneldense_1/biasoutput_dense/kerneloutput_dense/biasdecaylearning_ratemomentumweight_decay	SGDW/itertotal_1count_1totalcounttrue_positives_3false_negatives_2true_positives_2false_positives_2true_positives_1false_positives_1false_negatives_1weights_intermediatetrue_positivestrue_negativesfalse_positivesfalse_negativesSGDW/conv2d/kernel/momentumSGDW/conv2d/bias/momentumSGDW/conv2d_1/kernel/momentumSGDW/conv2d_1/bias/momentumSGDW/dense/kernel/momentum'SGDW/batch_normalization/gamma/momentum&SGDW/batch_normalization/beta/momentumSGDW/dense_1/kernel/momentumSGDW/dense_1/bias/momentum!SGDW/output_dense/kernel/momentumSGDW/output_dense/bias/momentum*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *+
f&R$
"__inference__traced_restore_607577��
�
�
4__inference_batch_normalization_layer_call_fn_607147

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605618p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_607134

inputs3
matmul_readvariableop_resource:���
identity��MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:�����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
1__inference_trainable_layers_layer_call_fn_606024
trainable_input"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
	unknown_3:���
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�1

unknown_11:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltrainable_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*-
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nametrainable_input
�
d
6__inference_adaptive_thresholding_layer_call_fn_606831
input_batch
unknown
identity�
PartitionedCallPartitionedCallinput_batchunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606163j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:�����������::^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_batch:,(
&
_output_shapes
:
�
�
F__inference_full_model_layer_call_and_return_conditional_losses_606464
preprocessing_input 
adaptive_thresholding_6064322
trainable_layers_606436:�&
trainable_layers_606438:	�3
trainable_layers_606440:��&
trainable_layers_606442:	�,
trainable_layers_606444:���&
trainable_layers_606446:	�&
trainable_layers_606448:	�&
trainable_layers_606450:	�&
trainable_layers_606452:	�+
trainable_layers_606454:
��&
trainable_layers_606456:	�*
trainable_layers_606458:	�1%
trainable_layers_606460:1
identity��(trainable_layers/StatefulPartitionedCall�
grayscale/PartitionedCallPartitionedCallpreprocessing_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_grayscale_layer_call_and_return_conditional_losses_606145�
%adaptive_thresholding/PartitionedCallPartitionedCall"grayscale/PartitionedCall:output:0adaptive_thresholding_606432*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606163�
rescaling/PartitionedCallPartitionedCall.adaptive_thresholding/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_606176�
(trainable_layers/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0trainable_layers_606436trainable_layers_606438trainable_layers_606440trainable_layers_606442trainable_layers_606444trainable_layers_606446trainable_layers_606448trainable_layers_606450trainable_layers_606452trainable_layers_606454trainable_layers_606456trainable_layers_606458trainable_layers_606460*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*-
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605964�
IdentityIdentity1trainable_layers/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1q
NoOpNoOp)^trainable_layers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2T
(trainable_layers/StatefulPartitionedCall(trainable_layers/StatefulPartitionedCall:f b
1
_output_shapes
:�����������
-
_user_specified_namepreprocessing_input:,(
&
_output_shapes
:
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_607109

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_full_model_layer_call_fn_606237
preprocessing_input
unknown$
	unknown_0:�
	unknown_1:	�%
	unknown_2:��
	unknown_3:	�
	unknown_4:���
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�1

unknown_12:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpreprocessing_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_full_model_layer_call_and_return_conditional_losses_606206o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:�����������
-
_user_specified_namepreprocessing_input:,(
&
_output_shapes
:
�
�
F__inference_full_model_layer_call_and_return_conditional_losses_606328

inputs 
adaptive_thresholding_6062962
trainable_layers_606300:�&
trainable_layers_606302:	�3
trainable_layers_606304:��&
trainable_layers_606306:	�,
trainable_layers_606308:���&
trainable_layers_606310:	�&
trainable_layers_606312:	�&
trainable_layers_606314:	�&
trainable_layers_606316:	�+
trainable_layers_606318:
��&
trainable_layers_606320:	�*
trainable_layers_606322:	�1%
trainable_layers_606324:1
identity��(trainable_layers/StatefulPartitionedCall�
grayscale/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_grayscale_layer_call_and_return_conditional_losses_606145�
%adaptive_thresholding/PartitionedCallPartitionedCall"grayscale/PartitionedCall:output:0adaptive_thresholding_606296*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606163�
rescaling/PartitionedCallPartitionedCall.adaptive_thresholding/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_606176�
(trainable_layers/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0trainable_layers_606300trainable_layers_606302trainable_layers_606304trainable_layers_606306trainable_layers_606308trainable_layers_606310trainable_layers_606312trainable_layers_606314trainable_layers_606316trainable_layers_606318trainable_layers_606320trainable_layers_606322trainable_layers_606324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*-
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605964�
IdentityIdentity1trainable_layers/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1q
NoOpNoOp)^trainable_layers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2T
(trainable_layers/StatefulPartitionedCall(trainable_layers/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:,(
&
_output_shapes
:
�	
�
H__inference_output_dense_layer_call_and_return_conditional_losses_607263

inputs1
matmul_readvariableop_resource:	�1-
biasadd_readvariableop_resource:1
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������1w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_output_dense_layer_call_fn_607253

inputs
unknown:	�1
	unknown_0:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_output_dense_layer_call_and_return_conditional_losses_605781o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_607069

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������[
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:������������l
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
B
&__inference_re_lu_layer_call_fn_607219

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_605752a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
1__inference_trainable_layers_layer_call_fn_606892

inputs"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
	unknown_3:���
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�1

unknown_11:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605795o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
ܳ
�
F__inference_full_model_layer_call_and_return_conditional_losses_606787

inputs'
#adaptive_thresholding_conv2d_filterQ
6trainable_layers_conv2d_conv2d_readvariableop_resource:�F
7trainable_layers_conv2d_biasadd_readvariableop_resource:	�T
8trainable_layers_conv2d_1_conv2d_readvariableop_resource:��H
9trainable_layers_conv2d_1_biasadd_readvariableop_resource:	�J
5trainable_layers_dense_matmul_readvariableop_resource:���[
Ltrainable_layers_batch_normalization_assignmovingavg_readvariableop_resource:	�]
Ntrainable_layers_batch_normalization_assignmovingavg_1_readvariableop_resource:	�P
Atrainable_layers_batch_normalization_cast_readvariableop_resource:	�R
Ctrainable_layers_batch_normalization_cast_1_readvariableop_resource:	�K
7trainable_layers_dense_1_matmul_readvariableop_resource:
��G
8trainable_layers_dense_1_biasadd_readvariableop_resource:	�O
<trainable_layers_output_dense_matmul_readvariableop_resource:	�1K
=trainable_layers_output_dense_biasadd_readvariableop_resource:1
identity��4trainable_layers/batch_normalization/AssignMovingAvg�Ctrainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOp�6trainable_layers/batch_normalization/AssignMovingAvg_1�Etrainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOp�8trainable_layers/batch_normalization/Cast/ReadVariableOp�:trainable_layers/batch_normalization/Cast_1/ReadVariableOp�.trainable_layers/conv2d/BiasAdd/ReadVariableOp�-trainable_layers/conv2d/Conv2D/ReadVariableOp�0trainable_layers/conv2d_1/BiasAdd/ReadVariableOp�/trainable_layers/conv2d_1/Conv2D/ReadVariableOp�,trainable_layers/dense/MatMul/ReadVariableOp�/trainable_layers/dense_1/BiasAdd/ReadVariableOp�.trainable_layers/dense_1/MatMul/ReadVariableOp�4trainable_layers/output_dense/BiasAdd/ReadVariableOp�3trainable_layers/output_dense/MatMul/ReadVariableOps
#grayscale/rgb_to_grayscale/IdentityIdentityinputs*
T0*1
_output_shapes
:�����������{
&grayscale/rgb_to_grayscale/Tensordot/bConst*
_output_shapes
:*
dtype0*!
valueB"l	�>�E?�x�=s
)grayscale/rgb_to_grayscale/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
)grayscale/rgb_to_grayscale/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          �
*grayscale/rgb_to_grayscale/Tensordot/ShapeShape,grayscale/rgb_to_grayscale/Identity:output:0*
T0*
_output_shapes
:t
2grayscale/rgb_to_grayscale/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-grayscale/rgb_to_grayscale/Tensordot/GatherV2GatherV23grayscale/rgb_to_grayscale/Tensordot/Shape:output:02grayscale/rgb_to_grayscale/Tensordot/free:output:0;grayscale/rgb_to_grayscale/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4grayscale/rgb_to_grayscale/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/grayscale/rgb_to_grayscale/Tensordot/GatherV2_1GatherV23grayscale/rgb_to_grayscale/Tensordot/Shape:output:02grayscale/rgb_to_grayscale/Tensordot/axes:output:0=grayscale/rgb_to_grayscale/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*grayscale/rgb_to_grayscale/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
)grayscale/rgb_to_grayscale/Tensordot/ProdProd6grayscale/rgb_to_grayscale/Tensordot/GatherV2:output:03grayscale/rgb_to_grayscale/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,grayscale/rgb_to_grayscale/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
+grayscale/rgb_to_grayscale/Tensordot/Prod_1Prod8grayscale/rgb_to_grayscale/Tensordot/GatherV2_1:output:05grayscale/rgb_to_grayscale/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0grayscale/rgb_to_grayscale/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+grayscale/rgb_to_grayscale/Tensordot/concatConcatV22grayscale/rgb_to_grayscale/Tensordot/free:output:02grayscale/rgb_to_grayscale/Tensordot/axes:output:09grayscale/rgb_to_grayscale/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
*grayscale/rgb_to_grayscale/Tensordot/stackPack2grayscale/rgb_to_grayscale/Tensordot/Prod:output:04grayscale/rgb_to_grayscale/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
.grayscale/rgb_to_grayscale/Tensordot/transpose	Transpose,grayscale/rgb_to_grayscale/Identity:output:04grayscale/rgb_to_grayscale/Tensordot/concat:output:0*
T0*1
_output_shapes
:������������
,grayscale/rgb_to_grayscale/Tensordot/ReshapeReshape2grayscale/rgb_to_grayscale/Tensordot/transpose:y:03grayscale/rgb_to_grayscale/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
4grayscale/rgb_to_grayscale/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
.grayscale/rgb_to_grayscale/Tensordot/Reshape_1Reshape/grayscale/rgb_to_grayscale/Tensordot/b:output:0=grayscale/rgb_to_grayscale/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
+grayscale/rgb_to_grayscale/Tensordot/MatMulMatMul5grayscale/rgb_to_grayscale/Tensordot/Reshape:output:07grayscale/rgb_to_grayscale/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������o
,grayscale/rgb_to_grayscale/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB t
2grayscale/rgb_to_grayscale/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-grayscale/rgb_to_grayscale/Tensordot/concat_1ConcatV26grayscale/rgb_to_grayscale/Tensordot/GatherV2:output:05grayscale/rgb_to_grayscale/Tensordot/Const_2:output:0;grayscale/rgb_to_grayscale/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
$grayscale/rgb_to_grayscale/TensordotReshape5grayscale/rgb_to_grayscale/Tensordot/MatMul:product:06grayscale/rgb_to_grayscale/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:�����������t
)grayscale/rgb_to_grayscale/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%grayscale/rgb_to_grayscale/ExpandDims
ExpandDims-grayscale/rgb_to_grayscale/Tensordot:output:02grayscale/rgb_to_grayscale/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
grayscale/rgb_to_grayscaleIdentity.grayscale/rgb_to_grayscale/ExpandDims:output:0*
T0*1
_output_shapes
:������������
adaptive_thresholding/Conv2DConv2D#grayscale/rgb_to_grayscale:output:0#adaptive_thresholding_conv2d_filter*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
`
adaptive_thresholding/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
adaptive_thresholding/subSub%adaptive_thresholding/Conv2D:output:0$adaptive_thresholding/sub/y:output:0*
T0*1
_output_shapes
:������������
adaptive_thresholding/LessEqual	LessEqual#grayscale/rgb_to_grayscale:output:0adaptive_thresholding/sub:z:0*
T0*1
_output_shapes
:�����������}
$adaptive_thresholding/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
adaptive_thresholding/transpose	Transpose#adaptive_thresholding/LessEqual:z:0-adaptive_thresholding/transpose/perm:output:0*
T0
*1
_output_shapes
:�����������c
 adaptive_thresholding/SelectV2/tConst*
_output_shapes
: *
dtype0*
value
B :�b
 adaptive_thresholding/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : �
adaptive_thresholding/SelectV2SelectV2#adaptive_thresholding/transpose:y:0)adaptive_thresholding/SelectV2/t:output:0)adaptive_thresholding/SelectV2/e:output:0*
T0*1
_output_shapes
:�����������
&adaptive_thresholding/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
!adaptive_thresholding/transpose_1	Transpose'adaptive_thresholding/SelectV2:output:0/adaptive_thresholding/transpose_1/perm:output:0*
T0*1
_output_shapes
:�����������U
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
rescaling/Cast_2Cast%adaptive_thresholding/transpose_1:y:0*

DstT0*

SrcT0*1
_output_shapes
:������������
rescaling/mulMulrescaling/Cast_2:y:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:������������
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:������������
-trainable_layers/conv2d/Conv2D/ReadVariableOpReadVariableOp6trainable_layers_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
trainable_layers/conv2d/Conv2DConv2Drescaling/add:z:05trainable_layers/conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
�
.trainable_layers/conv2d/BiasAdd/ReadVariableOpReadVariableOp7trainable_layers_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
trainable_layers/conv2d/BiasAddBiasAdd'trainable_layers/conv2d/Conv2D:output:06trainable_layers/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
trainable_layers/conv2d/ReluRelu(trainable_layers/conv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:�������������
&trainable_layers/max_pooling2d/MaxPoolMaxPool*trainable_layers/conv2d/Relu:activations:0*0
_output_shapes
:���������00�*
ksize
*
paddingVALID*
strides
�
/trainable_layers/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8trainable_layers_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 trainable_layers/conv2d_1/Conv2DConv2D/trainable_layers/max_pooling2d/MaxPool:output:07trainable_layers/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
�
0trainable_layers/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9trainable_layers_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!trainable_layers/conv2d_1/BiasAddBiasAdd)trainable_layers/conv2d_1/Conv2D:output:08trainable_layers/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,��
trainable_layers/conv2d_1/ReluRelu*trainable_layers/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������,,��
(trainable_layers/max_pooling2d_1/MaxPoolMaxPool,trainable_layers/conv2d_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
o
trainable_layers/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  �
 trainable_layers/flatten/ReshapeReshape1trainable_layers/max_pooling2d_1/MaxPool:output:0'trainable_layers/flatten/Const:output:0*
T0*)
_output_shapes
:������������
,trainable_layers/dense/MatMul/ReadVariableOpReadVariableOp5trainable_layers_dense_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
trainable_layers/dense/MatMulMatMul)trainable_layers/flatten/Reshape:output:04trainable_layers/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Ctrainable_layers/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
1trainable_layers/batch_normalization/moments/meanMean'trainable_layers/dense/MatMul:product:0Ltrainable_layers/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
9trainable_layers/batch_normalization/moments/StopGradientStopGradient:trainable_layers/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
>trainable_layers/batch_normalization/moments/SquaredDifferenceSquaredDifference'trainable_layers/dense/MatMul:product:0Btrainable_layers/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
Gtrainable_layers/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
5trainable_layers/batch_normalization/moments/varianceMeanBtrainable_layers/batch_normalization/moments/SquaredDifference:z:0Ptrainable_layers/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
4trainable_layers/batch_normalization/moments/SqueezeSqueeze:trainable_layers/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
6trainable_layers/batch_normalization/moments/Squeeze_1Squeeze>trainable_layers/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 
:trainable_layers/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Ctrainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpLtrainable_layers_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8trainable_layers/batch_normalization/AssignMovingAvg/subSubKtrainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOp:value:0=trainable_layers/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
8trainable_layers/batch_normalization/AssignMovingAvg/mulMul<trainable_layers/batch_normalization/AssignMovingAvg/sub:z:0Ctrainable_layers/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/AssignMovingAvgAssignSubVariableOpLtrainable_layers_batch_normalization_assignmovingavg_readvariableop_resource<trainable_layers/batch_normalization/AssignMovingAvg/mul:z:0D^trainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
<trainable_layers/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Etrainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpNtrainable_layers_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:trainable_layers/batch_normalization/AssignMovingAvg_1/subSubMtrainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0?trainable_layers/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
:trainable_layers/batch_normalization/AssignMovingAvg_1/mulMul>trainable_layers/batch_normalization/AssignMovingAvg_1/sub:z:0Etrainable_layers/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
6trainable_layers/batch_normalization/AssignMovingAvg_1AssignSubVariableOpNtrainable_layers_batch_normalization_assignmovingavg_1_readvariableop_resource>trainable_layers/batch_normalization/AssignMovingAvg_1/mul:z:0F^trainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
8trainable_layers/batch_normalization/Cast/ReadVariableOpReadVariableOpAtrainable_layers_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:trainable_layers/batch_normalization/Cast_1/ReadVariableOpReadVariableOpCtrainable_layers_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4trainable_layers/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2trainable_layers/batch_normalization/batchnorm/addAddV2?trainable_layers/batch_normalization/moments/Squeeze_1:output:0=trainable_layers/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/batchnorm/RsqrtRsqrt6trainable_layers/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2trainable_layers/batch_normalization/batchnorm/mulMul8trainable_layers/batch_normalization/batchnorm/Rsqrt:y:0Btrainable_layers/batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/batchnorm/mul_1Mul'trainable_layers/dense/MatMul:product:06trainable_layers/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4trainable_layers/batch_normalization/batchnorm/mul_2Mul=trainable_layers/batch_normalization/moments/Squeeze:output:06trainable_layers/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2trainable_layers/batch_normalization/batchnorm/subSub@trainable_layers/batch_normalization/Cast/ReadVariableOp:value:08trainable_layers/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/batchnorm/add_1AddV28trainable_layers/batch_normalization/batchnorm/mul_1:z:06trainable_layers/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
trainable_layers/re_lu/ReluRelu8trainable_layers/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
.trainable_layers/dense_1/MatMul/ReadVariableOpReadVariableOp7trainable_layers_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
trainable_layers/dense_1/MatMulMatMul)trainable_layers/re_lu/Relu:activations:06trainable_layers/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/trainable_layers/dense_1/BiasAdd/ReadVariableOpReadVariableOp8trainable_layers_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 trainable_layers/dense_1/BiasAddBiasAdd)trainable_layers/dense_1/MatMul:product:07trainable_layers/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
trainable_layers/dense_1/ReluRelu)trainable_layers/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
3trainable_layers/output_dense/MatMul/ReadVariableOpReadVariableOp<trainable_layers_output_dense_matmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0�
$trainable_layers/output_dense/MatMulMatMul+trainable_layers/dense_1/Relu:activations:0;trainable_layers/output_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
4trainable_layers/output_dense/BiasAdd/ReadVariableOpReadVariableOp=trainable_layers_output_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
%trainable_layers/output_dense/BiasAddBiasAdd.trainable_layers/output_dense/MatMul:product:0<trainable_layers/output_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
'trainable_layers/softmax_output/SoftmaxSoftmax.trainable_layers/output_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������1�
IdentityIdentity1trainable_layers/softmax_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp5^trainable_layers/batch_normalization/AssignMovingAvgD^trainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOp7^trainable_layers/batch_normalization/AssignMovingAvg_1F^trainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOp9^trainable_layers/batch_normalization/Cast/ReadVariableOp;^trainable_layers/batch_normalization/Cast_1/ReadVariableOp/^trainable_layers/conv2d/BiasAdd/ReadVariableOp.^trainable_layers/conv2d/Conv2D/ReadVariableOp1^trainable_layers/conv2d_1/BiasAdd/ReadVariableOp0^trainable_layers/conv2d_1/Conv2D/ReadVariableOp-^trainable_layers/dense/MatMul/ReadVariableOp0^trainable_layers/dense_1/BiasAdd/ReadVariableOp/^trainable_layers/dense_1/MatMul/ReadVariableOp5^trainable_layers/output_dense/BiasAdd/ReadVariableOp4^trainable_layers/output_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2l
4trainable_layers/batch_normalization/AssignMovingAvg4trainable_layers/batch_normalization/AssignMovingAvg2�
Ctrainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOpCtrainable_layers/batch_normalization/AssignMovingAvg/ReadVariableOp2p
6trainable_layers/batch_normalization/AssignMovingAvg_16trainable_layers/batch_normalization/AssignMovingAvg_12�
Etrainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOpEtrainable_layers/batch_normalization/AssignMovingAvg_1/ReadVariableOp2t
8trainable_layers/batch_normalization/Cast/ReadVariableOp8trainable_layers/batch_normalization/Cast/ReadVariableOp2x
:trainable_layers/batch_normalization/Cast_1/ReadVariableOp:trainable_layers/batch_normalization/Cast_1/ReadVariableOp2`
.trainable_layers/conv2d/BiasAdd/ReadVariableOp.trainable_layers/conv2d/BiasAdd/ReadVariableOp2^
-trainable_layers/conv2d/Conv2D/ReadVariableOp-trainable_layers/conv2d/Conv2D/ReadVariableOp2d
0trainable_layers/conv2d_1/BiasAdd/ReadVariableOp0trainable_layers/conv2d_1/BiasAdd/ReadVariableOp2b
/trainable_layers/conv2d_1/Conv2D/ReadVariableOp/trainable_layers/conv2d_1/Conv2D/ReadVariableOp2\
,trainable_layers/dense/MatMul/ReadVariableOp,trainable_layers/dense/MatMul/ReadVariableOp2b
/trainable_layers/dense_1/BiasAdd/ReadVariableOp/trainable_layers/dense_1/BiasAdd/ReadVariableOp2`
.trainable_layers/dense_1/MatMul/ReadVariableOp.trainable_layers/dense_1/MatMul/ReadVariableOp2l
4trainable_layers/output_dense/BiasAdd/ReadVariableOp4trainable_layers/output_dense/BiasAdd/ReadVariableOp2j
3trainable_layers/output_dense/MatMul/ReadVariableOp3trainable_layers/output_dense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:,(
&
_output_shapes
:
�
F
*__inference_rescaling_layer_call_fn_606852

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_606176j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_607577
file_prefix9
assignvariableop_conv2d_kernel:�-
assignvariableop_1_conv2d_bias:	�>
"assignvariableop_2_conv2d_1_kernel:��/
 assignvariableop_3_conv2d_1_bias:	�4
assignvariableop_4_dense_kernel:���;
,assignvariableop_5_batch_normalization_gamma:	�:
+assignvariableop_6_batch_normalization_beta:	�A
2assignvariableop_7_batch_normalization_moving_mean:	�E
6assignvariableop_8_batch_normalization_moving_variance:	�5
!assignvariableop_9_dense_1_kernel:
��/
 assignvariableop_10_dense_1_bias:	�:
'assignvariableop_11_output_dense_kernel:	�13
%assignvariableop_12_output_dense_bias:1#
assignvariableop_13_decay: +
!assignvariableop_14_learning_rate: &
assignvariableop_15_momentum: *
 assignvariableop_16_weight_decay: '
assignvariableop_17_sgdw_iter:	 %
assignvariableop_18_total_1: %
assignvariableop_19_count_1: #
assignvariableop_20_total: #
assignvariableop_21_count: 2
$assignvariableop_22_true_positives_3:3
%assignvariableop_23_false_negatives_2:2
$assignvariableop_24_true_positives_2:3
%assignvariableop_25_false_positives_2:2
$assignvariableop_26_true_positives_1:13
%assignvariableop_27_false_positives_1:13
%assignvariableop_28_false_negatives_1:16
(assignvariableop_29_weights_intermediate:15
"assignvariableop_30_true_positives:	�15
"assignvariableop_31_true_negatives:	�16
#assignvariableop_32_false_positives:	�16
#assignvariableop_33_false_negatives:	�1J
/assignvariableop_34_sgdw_conv2d_kernel_momentum:�<
-assignvariableop_35_sgdw_conv2d_bias_momentum:	�M
1assignvariableop_36_sgdw_conv2d_1_kernel_momentum:��>
/assignvariableop_37_sgdw_conv2d_1_bias_momentum:	�C
.assignvariableop_38_sgdw_dense_kernel_momentum:���J
;assignvariableop_39_sgdw_batch_normalization_gamma_momentum:	�I
:assignvariableop_40_sgdw_batch_normalization_beta_momentum:	�D
0assignvariableop_41_sgdw_dense_1_kernel_momentum:
��=
.assignvariableop_42_sgdw_dense_1_bias_momentum:	�H
5assignvariableop_43_sgdw_output_dense_kernel_momentum:	�1A
3assignvariableop_44_sgdw_output_dense_bias_momentum:1
identity_46��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*�
value�B�.B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/4/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_batch_normalization_gammaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp+assignvariableop_6_batch_normalization_betaIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp2assignvariableop_7_batch_normalization_moving_meanIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_moving_varianceIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_1_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_dense_1_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_output_dense_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_output_dense_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_decayIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp!assignvariableop_14_learning_rateIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_momentumIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp assignvariableop_16_weight_decayIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_sgdw_iterIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_total_1Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_count_1Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_true_positives_3Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp%assignvariableop_23_false_negatives_2Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp$assignvariableop_24_true_positives_2Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp%assignvariableop_25_false_positives_2Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_true_positives_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp%assignvariableop_27_false_positives_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_false_negatives_1Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_weights_intermediateIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_true_positivesIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp"assignvariableop_31_true_negativesIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp#assignvariableop_32_false_positivesIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp#assignvariableop_33_false_negativesIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp/assignvariableop_34_sgdw_conv2d_kernel_momentumIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp-assignvariableop_35_sgdw_conv2d_bias_momentumIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp1assignvariableop_36_sgdw_conv2d_1_kernel_momentumIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp/assignvariableop_37_sgdw_conv2d_1_bias_momentumIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp.assignvariableop_38_sgdw_dense_kernel_momentumIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp;assignvariableop_39_sgdw_batch_normalization_gamma_momentumIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp:assignvariableop_40_sgdw_batch_normalization_beta_momentumIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp0assignvariableop_41_sgdw_dense_1_kernel_momentumIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp.assignvariableop_42_sgdw_dense_1_bias_momentumIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp5assignvariableop_43_sgdw_output_dense_kernel_momentumIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp3assignvariableop_44_sgdw_output_dense_bias_momentumIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
)__inference_conv2d_1_layer_call_fn_607088

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������,,�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_605712x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������,,�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������00�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�$
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605665

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606847
input_batch
conv2d_filter
identity�
Conv2DConv2Dinput_batchconv2d_filter*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@g
subSubConv2D:output:0sub/y:output:0*
T0*1
_output_shapes
:�����������h
	LessEqual	LessEqualinput_batchsub:z:0*
T0*1
_output_shapes
:�����������g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             z
	transpose	TransposeLessEqual:z:0transpose/perm:output:0*
T0
*1
_output_shapes
:�����������M

SelectV2/tConst*
_output_shapes
: *
dtype0*
value
B :�L

SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : �
SelectV2SelectV2transpose:y:0SelectV2/t:output:0SelectV2/e:output:0*
T0*1
_output_shapes
:�����������i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_1	TransposeSelectV2:output:0transpose_1/perm:output:0*
T0*1
_output_shapes
:�����������a
IdentityIdentitytranspose_1:y:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:�����������::^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_batch:,(
&
_output_shapes
:
�
D
(__inference_flatten_layer_call_fn_607114

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_605725b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_605591

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
K
/__inference_softmax_output_layer_call_fn_607268

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_softmax_output_layer_call_and_return_conditional_losses_605792`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������1:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�
�
F__inference_full_model_layer_call_and_return_conditional_losses_606672

inputs'
#adaptive_thresholding_conv2d_filterQ
6trainable_layers_conv2d_conv2d_readvariableop_resource:�F
7trainable_layers_conv2d_biasadd_readvariableop_resource:	�T
8trainable_layers_conv2d_1_conv2d_readvariableop_resource:��H
9trainable_layers_conv2d_1_biasadd_readvariableop_resource:	�J
5trainable_layers_dense_matmul_readvariableop_resource:���P
Atrainable_layers_batch_normalization_cast_readvariableop_resource:	�R
Ctrainable_layers_batch_normalization_cast_1_readvariableop_resource:	�R
Ctrainable_layers_batch_normalization_cast_2_readvariableop_resource:	�R
Ctrainable_layers_batch_normalization_cast_3_readvariableop_resource:	�K
7trainable_layers_dense_1_matmul_readvariableop_resource:
��G
8trainable_layers_dense_1_biasadd_readvariableop_resource:	�O
<trainable_layers_output_dense_matmul_readvariableop_resource:	�1K
=trainable_layers_output_dense_biasadd_readvariableop_resource:1
identity��8trainable_layers/batch_normalization/Cast/ReadVariableOp�:trainable_layers/batch_normalization/Cast_1/ReadVariableOp�:trainable_layers/batch_normalization/Cast_2/ReadVariableOp�:trainable_layers/batch_normalization/Cast_3/ReadVariableOp�.trainable_layers/conv2d/BiasAdd/ReadVariableOp�-trainable_layers/conv2d/Conv2D/ReadVariableOp�0trainable_layers/conv2d_1/BiasAdd/ReadVariableOp�/trainable_layers/conv2d_1/Conv2D/ReadVariableOp�,trainable_layers/dense/MatMul/ReadVariableOp�/trainable_layers/dense_1/BiasAdd/ReadVariableOp�.trainable_layers/dense_1/MatMul/ReadVariableOp�4trainable_layers/output_dense/BiasAdd/ReadVariableOp�3trainable_layers/output_dense/MatMul/ReadVariableOps
#grayscale/rgb_to_grayscale/IdentityIdentityinputs*
T0*1
_output_shapes
:�����������{
&grayscale/rgb_to_grayscale/Tensordot/bConst*
_output_shapes
:*
dtype0*!
valueB"l	�>�E?�x�=s
)grayscale/rgb_to_grayscale/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:~
)grayscale/rgb_to_grayscale/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          �
*grayscale/rgb_to_grayscale/Tensordot/ShapeShape,grayscale/rgb_to_grayscale/Identity:output:0*
T0*
_output_shapes
:t
2grayscale/rgb_to_grayscale/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-grayscale/rgb_to_grayscale/Tensordot/GatherV2GatherV23grayscale/rgb_to_grayscale/Tensordot/Shape:output:02grayscale/rgb_to_grayscale/Tensordot/free:output:0;grayscale/rgb_to_grayscale/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4grayscale/rgb_to_grayscale/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
/grayscale/rgb_to_grayscale/Tensordot/GatherV2_1GatherV23grayscale/rgb_to_grayscale/Tensordot/Shape:output:02grayscale/rgb_to_grayscale/Tensordot/axes:output:0=grayscale/rgb_to_grayscale/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*grayscale/rgb_to_grayscale/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
)grayscale/rgb_to_grayscale/Tensordot/ProdProd6grayscale/rgb_to_grayscale/Tensordot/GatherV2:output:03grayscale/rgb_to_grayscale/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,grayscale/rgb_to_grayscale/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
+grayscale/rgb_to_grayscale/Tensordot/Prod_1Prod8grayscale/rgb_to_grayscale/Tensordot/GatherV2_1:output:05grayscale/rgb_to_grayscale/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0grayscale/rgb_to_grayscale/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+grayscale/rgb_to_grayscale/Tensordot/concatConcatV22grayscale/rgb_to_grayscale/Tensordot/free:output:02grayscale/rgb_to_grayscale/Tensordot/axes:output:09grayscale/rgb_to_grayscale/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
*grayscale/rgb_to_grayscale/Tensordot/stackPack2grayscale/rgb_to_grayscale/Tensordot/Prod:output:04grayscale/rgb_to_grayscale/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
.grayscale/rgb_to_grayscale/Tensordot/transpose	Transpose,grayscale/rgb_to_grayscale/Identity:output:04grayscale/rgb_to_grayscale/Tensordot/concat:output:0*
T0*1
_output_shapes
:������������
,grayscale/rgb_to_grayscale/Tensordot/ReshapeReshape2grayscale/rgb_to_grayscale/Tensordot/transpose:y:03grayscale/rgb_to_grayscale/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
4grayscale/rgb_to_grayscale/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
.grayscale/rgb_to_grayscale/Tensordot/Reshape_1Reshape/grayscale/rgb_to_grayscale/Tensordot/b:output:0=grayscale/rgb_to_grayscale/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
+grayscale/rgb_to_grayscale/Tensordot/MatMulMatMul5grayscale/rgb_to_grayscale/Tensordot/Reshape:output:07grayscale/rgb_to_grayscale/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������o
,grayscale/rgb_to_grayscale/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB t
2grayscale/rgb_to_grayscale/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
-grayscale/rgb_to_grayscale/Tensordot/concat_1ConcatV26grayscale/rgb_to_grayscale/Tensordot/GatherV2:output:05grayscale/rgb_to_grayscale/Tensordot/Const_2:output:0;grayscale/rgb_to_grayscale/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
$grayscale/rgb_to_grayscale/TensordotReshape5grayscale/rgb_to_grayscale/Tensordot/MatMul:product:06grayscale/rgb_to_grayscale/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:�����������t
)grayscale/rgb_to_grayscale/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
%grayscale/rgb_to_grayscale/ExpandDims
ExpandDims-grayscale/rgb_to_grayscale/Tensordot:output:02grayscale/rgb_to_grayscale/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
grayscale/rgb_to_grayscaleIdentity.grayscale/rgb_to_grayscale/ExpandDims:output:0*
T0*1
_output_shapes
:������������
adaptive_thresholding/Conv2DConv2D#grayscale/rgb_to_grayscale:output:0#adaptive_thresholding_conv2d_filter*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
`
adaptive_thresholding/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
adaptive_thresholding/subSub%adaptive_thresholding/Conv2D:output:0$adaptive_thresholding/sub/y:output:0*
T0*1
_output_shapes
:������������
adaptive_thresholding/LessEqual	LessEqual#grayscale/rgb_to_grayscale:output:0adaptive_thresholding/sub:z:0*
T0*1
_output_shapes
:�����������}
$adaptive_thresholding/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
adaptive_thresholding/transpose	Transpose#adaptive_thresholding/LessEqual:z:0-adaptive_thresholding/transpose/perm:output:0*
T0
*1
_output_shapes
:�����������c
 adaptive_thresholding/SelectV2/tConst*
_output_shapes
: *
dtype0*
value
B :�b
 adaptive_thresholding/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : �
adaptive_thresholding/SelectV2SelectV2#adaptive_thresholding/transpose:y:0)adaptive_thresholding/SelectV2/t:output:0)adaptive_thresholding/SelectV2/e:output:0*
T0*1
_output_shapes
:�����������
&adaptive_thresholding/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
!adaptive_thresholding/transpose_1	Transpose'adaptive_thresholding/SelectV2:output:0/adaptive_thresholding/transpose_1/perm:output:0*
T0*1
_output_shapes
:�����������U
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
rescaling/Cast_2Cast%adaptive_thresholding/transpose_1:y:0*

DstT0*

SrcT0*1
_output_shapes
:������������
rescaling/mulMulrescaling/Cast_2:y:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:������������
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:������������
-trainable_layers/conv2d/Conv2D/ReadVariableOpReadVariableOp6trainable_layers_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
trainable_layers/conv2d/Conv2DConv2Drescaling/add:z:05trainable_layers/conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
�
.trainable_layers/conv2d/BiasAdd/ReadVariableOpReadVariableOp7trainable_layers_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
trainable_layers/conv2d/BiasAddBiasAdd'trainable_layers/conv2d/Conv2D:output:06trainable_layers/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
trainable_layers/conv2d/ReluRelu(trainable_layers/conv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:�������������
&trainable_layers/max_pooling2d/MaxPoolMaxPool*trainable_layers/conv2d/Relu:activations:0*0
_output_shapes
:���������00�*
ksize
*
paddingVALID*
strides
�
/trainable_layers/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8trainable_layers_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 trainable_layers/conv2d_1/Conv2DConv2D/trainable_layers/max_pooling2d/MaxPool:output:07trainable_layers/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
�
0trainable_layers/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9trainable_layers_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!trainable_layers/conv2d_1/BiasAddBiasAdd)trainable_layers/conv2d_1/Conv2D:output:08trainable_layers/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,��
trainable_layers/conv2d_1/ReluRelu*trainable_layers/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������,,��
(trainable_layers/max_pooling2d_1/MaxPoolMaxPool,trainable_layers/conv2d_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
o
trainable_layers/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  �
 trainable_layers/flatten/ReshapeReshape1trainable_layers/max_pooling2d_1/MaxPool:output:0'trainable_layers/flatten/Const:output:0*
T0*)
_output_shapes
:������������
,trainable_layers/dense/MatMul/ReadVariableOpReadVariableOp5trainable_layers_dense_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
trainable_layers/dense/MatMulMatMul)trainable_layers/flatten/Reshape:output:04trainable_layers/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
8trainable_layers/batch_normalization/Cast/ReadVariableOpReadVariableOpAtrainable_layers_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:trainable_layers/batch_normalization/Cast_1/ReadVariableOpReadVariableOpCtrainable_layers_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:trainable_layers/batch_normalization/Cast_2/ReadVariableOpReadVariableOpCtrainable_layers_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:trainable_layers/batch_normalization/Cast_3/ReadVariableOpReadVariableOpCtrainable_layers_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0y
4trainable_layers/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2trainable_layers/batch_normalization/batchnorm/addAddV2Btrainable_layers/batch_normalization/Cast_1/ReadVariableOp:value:0=trainable_layers/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/batchnorm/RsqrtRsqrt6trainable_layers/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2trainable_layers/batch_normalization/batchnorm/mulMul8trainable_layers/batch_normalization/batchnorm/Rsqrt:y:0Btrainable_layers/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/batchnorm/mul_1Mul'trainable_layers/dense/MatMul:product:06trainable_layers/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4trainable_layers/batch_normalization/batchnorm/mul_2Mul@trainable_layers/batch_normalization/Cast/ReadVariableOp:value:06trainable_layers/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2trainable_layers/batch_normalization/batchnorm/subSubBtrainable_layers/batch_normalization/Cast_2/ReadVariableOp:value:08trainable_layers/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
4trainable_layers/batch_normalization/batchnorm/add_1AddV28trainable_layers/batch_normalization/batchnorm/mul_1:z:06trainable_layers/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
trainable_layers/re_lu/ReluRelu8trainable_layers/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
.trainable_layers/dense_1/MatMul/ReadVariableOpReadVariableOp7trainable_layers_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
trainable_layers/dense_1/MatMulMatMul)trainable_layers/re_lu/Relu:activations:06trainable_layers/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/trainable_layers/dense_1/BiasAdd/ReadVariableOpReadVariableOp8trainable_layers_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 trainable_layers/dense_1/BiasAddBiasAdd)trainable_layers/dense_1/MatMul:product:07trainable_layers/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
trainable_layers/dense_1/ReluRelu)trainable_layers/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
3trainable_layers/output_dense/MatMul/ReadVariableOpReadVariableOp<trainable_layers_output_dense_matmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0�
$trainable_layers/output_dense/MatMulMatMul+trainable_layers/dense_1/Relu:activations:0;trainable_layers/output_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
4trainable_layers/output_dense/BiasAdd/ReadVariableOpReadVariableOp=trainable_layers_output_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
%trainable_layers/output_dense/BiasAddBiasAdd.trainable_layers/output_dense/MatMul:product:0<trainable_layers/output_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
'trainable_layers/softmax_output/SoftmaxSoftmax.trainable_layers/output_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������1�
IdentityIdentity1trainable_layers/softmax_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp9^trainable_layers/batch_normalization/Cast/ReadVariableOp;^trainable_layers/batch_normalization/Cast_1/ReadVariableOp;^trainable_layers/batch_normalization/Cast_2/ReadVariableOp;^trainable_layers/batch_normalization/Cast_3/ReadVariableOp/^trainable_layers/conv2d/BiasAdd/ReadVariableOp.^trainable_layers/conv2d/Conv2D/ReadVariableOp1^trainable_layers/conv2d_1/BiasAdd/ReadVariableOp0^trainable_layers/conv2d_1/Conv2D/ReadVariableOp-^trainable_layers/dense/MatMul/ReadVariableOp0^trainable_layers/dense_1/BiasAdd/ReadVariableOp/^trainable_layers/dense_1/MatMul/ReadVariableOp5^trainable_layers/output_dense/BiasAdd/ReadVariableOp4^trainable_layers/output_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2t
8trainable_layers/batch_normalization/Cast/ReadVariableOp8trainable_layers/batch_normalization/Cast/ReadVariableOp2x
:trainable_layers/batch_normalization/Cast_1/ReadVariableOp:trainable_layers/batch_normalization/Cast_1/ReadVariableOp2x
:trainable_layers/batch_normalization/Cast_2/ReadVariableOp:trainable_layers/batch_normalization/Cast_2/ReadVariableOp2x
:trainable_layers/batch_normalization/Cast_3/ReadVariableOp:trainable_layers/batch_normalization/Cast_3/ReadVariableOp2`
.trainable_layers/conv2d/BiasAdd/ReadVariableOp.trainable_layers/conv2d/BiasAdd/ReadVariableOp2^
-trainable_layers/conv2d/Conv2D/ReadVariableOp-trainable_layers/conv2d/Conv2D/ReadVariableOp2d
0trainable_layers/conv2d_1/BiasAdd/ReadVariableOp0trainable_layers/conv2d_1/BiasAdd/ReadVariableOp2b
/trainable_layers/conv2d_1/Conv2D/ReadVariableOp/trainable_layers/conv2d_1/Conv2D/ReadVariableOp2\
,trainable_layers/dense/MatMul/ReadVariableOp,trainable_layers/dense/MatMul/ReadVariableOp2b
/trainable_layers/dense_1/BiasAdd/ReadVariableOp/trainable_layers/dense_1/BiasAdd/ReadVariableOp2`
.trainable_layers/dense_1/MatMul/ReadVariableOp.trainable_layers/dense_1/MatMul/ReadVariableOp2l
4trainable_layers/output_dense/BiasAdd/ReadVariableOp4trainable_layers/output_dense/BiasAdd/ReadVariableOp2j
3trainable_layers/output_dense/MatMul/ReadVariableOp3trainable_layers/output_dense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:,(
&
_output_shapes
:
�
�
(__inference_dense_1_layer_call_fn_607233

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_605765p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_1_layer_call_fn_607104

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_605591�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
]
A__inference_re_lu_layer_call_and_return_conditional_losses_607224

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
J__inference_softmax_output_layer_call_and_return_conditional_losses_607273

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������1Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������1:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�
�
+__inference_full_model_layer_call_fn_606538

inputs
unknown$
	unknown_0:�
	unknown_1:	�%
	unknown_2:��
	unknown_3:	�
	unknown_4:���
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�1

unknown_12:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_full_model_layer_call_and_return_conditional_losses_606206o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:,(
&
_output_shapes
:
�
�
4__inference_batch_normalization_layer_call_fn_607160

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605665p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607180

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605618

inputs+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�-
cast_2_readvariableop_resource:	�-
cast_3_readvariableop_resource:	�
identity��Cast/ReadVariableOp�Cast_1/ReadVariableOp�Cast_2/ReadVariableOp�Cast_3/ReadVariableOpm
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:u
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������l
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�n
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_full_model_layer_call_and_return_conditional_losses_606206

inputs 
adaptive_thresholding_6061642
trainable_layers_606178:�&
trainable_layers_606180:	�3
trainable_layers_606182:��&
trainable_layers_606184:	�,
trainable_layers_606186:���&
trainable_layers_606188:	�&
trainable_layers_606190:	�&
trainable_layers_606192:	�&
trainable_layers_606194:	�+
trainable_layers_606196:
��&
trainable_layers_606198:	�*
trainable_layers_606200:	�1%
trainable_layers_606202:1
identity��(trainable_layers/StatefulPartitionedCall�
grayscale/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_grayscale_layer_call_and_return_conditional_losses_606145�
%adaptive_thresholding/PartitionedCallPartitionedCall"grayscale/PartitionedCall:output:0adaptive_thresholding_606164*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606163�
rescaling/PartitionedCallPartitionedCall.adaptive_thresholding/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_606176�
(trainable_layers/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0trainable_layers_606178trainable_layers_606180trainable_layers_606182trainable_layers_606184trainable_layers_606186trainable_layers_606188trainable_layers_606190trainable_layers_606192trainable_layers_606194trainable_layers_606196trainable_layers_606198trainable_layers_606200trainable_layers_606202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605795�
IdentityIdentity1trainable_layers/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1q
NoOpNoOp)^trainable_layers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2T
(trainable_layers/StatefulPartitionedCall(trainable_layers/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:,(
&
_output_shapes
:
�
K
*__inference_grayscale_layer_call_fn_606792
input_batch
identity�
PartitionedCallPartitionedCallinput_batch*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_grayscale_layer_call_and_return_conditional_losses_606145j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_batch
�
a
E__inference_rescaling_layer_call_and_return_conditional_losses_606861

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    a
Cast_2Castinputs*

DstT0*

SrcT0*1
_output_shapes
:�����������c
mulMul
Cast_2:y:0Cast/x:output:0*
T0*1
_output_shapes
:�����������d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:�����������Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_605725

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
&__inference_dense_layer_call_fn_607127

inputs
unknown:���
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_605734p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:�����������: 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_605579

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�1
�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605964

inputs(
conv2d_605926:�
conv2d_605928:	�+
conv2d_1_605932:��
conv2d_1_605934:	�!
dense_605939:���)
batch_normalization_605942:	�)
batch_normalization_605944:	�)
batch_normalization_605946:	�)
batch_normalization_605948:	�"
dense_1_605952:
��
dense_1_605954:	�&
output_dense_605957:	�1!
output_dense_605959:1
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�$output_dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_605926conv2d_605928*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_605694�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_605579�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_605932conv2d_1_605934*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������,,�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_605712�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_605591�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_605725�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_605939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_605734�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_605942batch_normalization_605944batch_normalization_605946batch_normalization_605948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605665�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_605752�
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_605952dense_1_605954*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_605765�
$output_dense/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_dense_605957output_dense_605959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_output_dense_layer_call_and_return_conditional_losses_605781�
softmax_output/PartitionedCallPartitionedCall-output_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_softmax_output_layer_call_and_return_conditional_losses_605792v
IdentityIdentity'softmax_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall%^output_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$output_dense/StatefulPartitionedCall$output_dense/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_607120

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
f
J__inference_softmax_output_layer_call_and_return_conditional_losses_605792

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:���������1Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:���������1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������1:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_607244

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
J
.__inference_max_pooling2d_layer_call_fn_607074

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_605579�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
a
E__inference_rescaling_layer_call_and_return_conditional_losses_606176

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    a
Cast_2Castinputs*

DstT0*

SrcT0*1
_output_shapes
:�����������c
mulMul
Cast_2:y:0Cast/x:output:0*
T0*1
_output_shapes
:�����������d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:�����������Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�Y
�
__inference__traced_save_607432
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop2
.savev2_output_dense_kernel_read_readvariableop0
,savev2_output_dense_bias_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop'
#savev2_momentum_read_readvariableop+
'savev2_weight_decay_read_readvariableop(
$savev2_sgdw_iter_read_readvariableop	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_true_positives_3_read_readvariableop0
,savev2_false_negatives_2_read_readvariableop/
+savev2_true_positives_2_read_readvariableop0
,savev2_false_positives_2_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop3
/savev2_weights_intermediate_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop:
6savev2_sgdw_conv2d_kernel_momentum_read_readvariableop8
4savev2_sgdw_conv2d_bias_momentum_read_readvariableop<
8savev2_sgdw_conv2d_1_kernel_momentum_read_readvariableop:
6savev2_sgdw_conv2d_1_bias_momentum_read_readvariableop9
5savev2_sgdw_dense_kernel_momentum_read_readvariableopF
Bsavev2_sgdw_batch_normalization_gamma_momentum_read_readvariableopE
Asavev2_sgdw_batch_normalization_beta_momentum_read_readvariableop;
7savev2_sgdw_dense_1_kernel_momentum_read_readvariableop9
5savev2_sgdw_dense_1_bias_momentum_read_readvariableop@
<savev2_sgdw_output_dense_kernel_momentum_read_readvariableop>
:savev2_sgdw_output_dense_bias_momentum_read_readvariableop
savev2_const_1

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*�
value�B�.B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBCkeras_api/metrics/4/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBIvariables/0/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/1/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/2/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/3/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/4/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/5/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/6/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBIvariables/9/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/10/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/11/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBJvariables/12/.OPTIMIZER_SLOT/optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop.savev2_output_dense_kernel_read_readvariableop,savev2_output_dense_bias_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop#savev2_momentum_read_readvariableop'savev2_weight_decay_read_readvariableop$savev2_sgdw_iter_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_true_positives_3_read_readvariableop,savev2_false_negatives_2_read_readvariableop+savev2_true_positives_2_read_readvariableop,savev2_false_positives_2_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop/savev2_weights_intermediate_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop6savev2_sgdw_conv2d_kernel_momentum_read_readvariableop4savev2_sgdw_conv2d_bias_momentum_read_readvariableop8savev2_sgdw_conv2d_1_kernel_momentum_read_readvariableop6savev2_sgdw_conv2d_1_bias_momentum_read_readvariableop5savev2_sgdw_dense_kernel_momentum_read_readvariableopBsavev2_sgdw_batch_normalization_gamma_momentum_read_readvariableopAsavev2_sgdw_batch_normalization_beta_momentum_read_readvariableop7savev2_sgdw_dense_1_kernel_momentum_read_readvariableop5savev2_sgdw_dense_1_bias_momentum_read_readvariableop<savev2_sgdw_output_dense_kernel_momentum_read_readvariableop:savev2_sgdw_output_dense_bias_momentum_read_readvariableopsavev2_const_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *<
dtypes2
02.	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :�:�:��:�:���:�:�:�:�:
��:�:	�1:1: : : : : : : : : :::::1:1:1:1:	�1:	�1:	�1:	�1:�:�:��:�:���:�:�:
��:�:	�1:1: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:'#
!
_output_shapes
:���:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!	

_output_shapes	
:�:&
"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�1: 

_output_shapes
:1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:1: 

_output_shapes
:1: 

_output_shapes
:1: 

_output_shapes
:1:%!

_output_shapes
:	�1:% !

_output_shapes
:	�1:%!!

_output_shapes
:	�1:%"!

_output_shapes
:	�1:-#)
'
_output_shapes
:�:!$

_output_shapes	
:�:.%*
(
_output_shapes
:��:!&

_output_shapes	
:�:''#
!
_output_shapes
:���:!(

_output_shapes	
:�:!)

_output_shapes	
:�:&*"
 
_output_shapes
:
��:!+

_output_shapes	
:�:%,!

_output_shapes
:	�1: -

_output_shapes
:1:.

_output_shapes
: 
�1
�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606106
trainable_input(
conv2d_606068:�
conv2d_606070:	�+
conv2d_1_606074:��
conv2d_1_606076:	�!
dense_606081:���)
batch_normalization_606084:	�)
batch_normalization_606086:	�)
batch_normalization_606088:	�)
batch_normalization_606090:	�"
dense_1_606094:
��
dense_1_606096:	�&
output_dense_606099:	�1!
output_dense_606101:1
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�$output_dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCalltrainable_inputconv2d_606068conv2d_606070*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_605694�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_605579�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_606074conv2d_1_606076*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������,,�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_605712�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_605591�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_605725�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_606081*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_605734�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_606084batch_normalization_606086batch_normalization_606088batch_normalization_606090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605665�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_605752�
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_606094dense_1_606096*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_605765�
$output_dense/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_dense_606099output_dense_606101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_output_dense_layer_call_and_return_conditional_losses_605781�
softmax_output/PartitionedCallPartitionedCall-output_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_softmax_output_layer_call_and_return_conditional_losses_605792v
IdentityIdentity'softmax_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall%^output_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$output_dense/StatefulPartitionedCall$output_dense/StatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nametrainable_input
�$
f
E__inference_grayscale_layer_call_and_return_conditional_losses_606824
input_batch
identityn
rgb_to_grayscale/IdentityIdentityinput_batch*
T0*1
_output_shapes
:�����������q
rgb_to_grayscale/Tensordot/bConst*
_output_shapes
:*
dtype0*!
valueB"l	�>�E?�x�=i
rgb_to_grayscale/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
rgb_to_grayscale/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          r
 rgb_to_grayscale/Tensordot/ShapeShape"rgb_to_grayscale/Identity:output:0*
T0*
_output_shapes
:j
(rgb_to_grayscale/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#rgb_to_grayscale/Tensordot/GatherV2GatherV2)rgb_to_grayscale/Tensordot/Shape:output:0(rgb_to_grayscale/Tensordot/free:output:01rgb_to_grayscale/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*rgb_to_grayscale/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%rgb_to_grayscale/Tensordot/GatherV2_1GatherV2)rgb_to_grayscale/Tensordot/Shape:output:0(rgb_to_grayscale/Tensordot/axes:output:03rgb_to_grayscale/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 rgb_to_grayscale/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
rgb_to_grayscale/Tensordot/ProdProd,rgb_to_grayscale/Tensordot/GatherV2:output:0)rgb_to_grayscale/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"rgb_to_grayscale/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!rgb_to_grayscale/Tensordot/Prod_1Prod.rgb_to_grayscale/Tensordot/GatherV2_1:output:0+rgb_to_grayscale/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&rgb_to_grayscale/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!rgb_to_grayscale/Tensordot/concatConcatV2(rgb_to_grayscale/Tensordot/free:output:0(rgb_to_grayscale/Tensordot/axes:output:0/rgb_to_grayscale/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 rgb_to_grayscale/Tensordot/stackPack(rgb_to_grayscale/Tensordot/Prod:output:0*rgb_to_grayscale/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$rgb_to_grayscale/Tensordot/transpose	Transpose"rgb_to_grayscale/Identity:output:0*rgb_to_grayscale/Tensordot/concat:output:0*
T0*1
_output_shapes
:������������
"rgb_to_grayscale/Tensordot/ReshapeReshape(rgb_to_grayscale/Tensordot/transpose:y:0)rgb_to_grayscale/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������{
*rgb_to_grayscale/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
$rgb_to_grayscale/Tensordot/Reshape_1Reshape%rgb_to_grayscale/Tensordot/b:output:03rgb_to_grayscale/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
!rgb_to_grayscale/Tensordot/MatMulMatMul+rgb_to_grayscale/Tensordot/Reshape:output:0-rgb_to_grayscale/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������e
"rgb_to_grayscale/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB j
(rgb_to_grayscale/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#rgb_to_grayscale/Tensordot/concat_1ConcatV2,rgb_to_grayscale/Tensordot/GatherV2:output:0+rgb_to_grayscale/Tensordot/Const_2:output:01rgb_to_grayscale/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
rgb_to_grayscale/TensordotReshape+rgb_to_grayscale/Tensordot/MatMul:product:0,rgb_to_grayscale/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:�����������j
rgb_to_grayscale/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rgb_to_grayscale/ExpandDims
ExpandDims#rgb_to_grayscale/Tensordot:output:0(rgb_to_grayscale/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������~
rgb_to_grayscaleIdentity$rgb_to_grayscale/ExpandDims:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityrgb_to_grayscale:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_batch
�
]
A__inference_re_lu_layer_call_and_return_conditional_losses_605752

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_full_model_layer_call_fn_606392
preprocessing_input
unknown$
	unknown_0:�
	unknown_1:	�%
	unknown_2:��
	unknown_3:	�
	unknown_4:���
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�1

unknown_12:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpreprocessing_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*-
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_full_model_layer_call_and_return_conditional_losses_606328o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:�����������
-
_user_specified_namepreprocessing_input:,(
&
_output_shapes
:
�
�
+__inference_full_model_layer_call_fn_606571

inputs
unknown$
	unknown_0:�
	unknown_1:	�%
	unknown_2:��
	unknown_3:	�
	unknown_4:���
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�1

unknown_12:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*-
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *O
fJRH
F__inference_full_model_layer_call_and_return_conditional_losses_606328o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs:,(
&
_output_shapes
:
�
�
$__inference_signature_wrapper_606505
preprocessing_input
unknown$
	unknown_0:�
	unknown_1:	�%
	unknown_2:��
	unknown_3:	�
	unknown_4:���
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�1

unknown_12:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpreprocessing_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� **
f%R#
!__inference__wrapped_model_605570o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:�����������
-
_user_specified_namepreprocessing_input:,(
&
_output_shapes
:
�
�
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606163
input_batch
conv2d_filter
identity�
Conv2DConv2Dinput_batchconv2d_filter*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@g
subSubConv2D:output:0sub/y:output:0*
T0*1
_output_shapes
:�����������h
	LessEqual	LessEqualinput_batchsub:z:0*
T0*1
_output_shapes
:�����������g
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             z
	transpose	TransposeLessEqual:z:0transpose/perm:output:0*
T0
*1
_output_shapes
:�����������M

SelectV2/tConst*
_output_shapes
: *
dtype0*
value
B :�L

SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : �
SelectV2SelectV2transpose:y:0SelectV2/t:output:0SelectV2/e:output:0*
T0*1
_output_shapes
:�����������i
transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
transpose_1	TransposeSelectV2:output:0transpose_1/perm:output:0*
T0*1
_output_shapes
:�����������a
IdentityIdentitytranspose_1:y:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:�����������::^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_batch:,(
&
_output_shapes
:
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_607099

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������,,�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������,,�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������00�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
�
!__inference__wrapped_model_605570
preprocessing_input2
.full_model_adaptive_thresholding_conv2d_filter\
Afull_model_trainable_layers_conv2d_conv2d_readvariableop_resource:�Q
Bfull_model_trainable_layers_conv2d_biasadd_readvariableop_resource:	�_
Cfull_model_trainable_layers_conv2d_1_conv2d_readvariableop_resource:��S
Dfull_model_trainable_layers_conv2d_1_biasadd_readvariableop_resource:	�U
@full_model_trainable_layers_dense_matmul_readvariableop_resource:���[
Lfull_model_trainable_layers_batch_normalization_cast_readvariableop_resource:	�]
Nfull_model_trainable_layers_batch_normalization_cast_1_readvariableop_resource:	�]
Nfull_model_trainable_layers_batch_normalization_cast_2_readvariableop_resource:	�]
Nfull_model_trainable_layers_batch_normalization_cast_3_readvariableop_resource:	�V
Bfull_model_trainable_layers_dense_1_matmul_readvariableop_resource:
��R
Cfull_model_trainable_layers_dense_1_biasadd_readvariableop_resource:	�Z
Gfull_model_trainable_layers_output_dense_matmul_readvariableop_resource:	�1V
Hfull_model_trainable_layers_output_dense_biasadd_readvariableop_resource:1
identity��Cfull_model/trainable_layers/batch_normalization/Cast/ReadVariableOp�Efull_model/trainable_layers/batch_normalization/Cast_1/ReadVariableOp�Efull_model/trainable_layers/batch_normalization/Cast_2/ReadVariableOp�Efull_model/trainable_layers/batch_normalization/Cast_3/ReadVariableOp�9full_model/trainable_layers/conv2d/BiasAdd/ReadVariableOp�8full_model/trainable_layers/conv2d/Conv2D/ReadVariableOp�;full_model/trainable_layers/conv2d_1/BiasAdd/ReadVariableOp�:full_model/trainable_layers/conv2d_1/Conv2D/ReadVariableOp�7full_model/trainable_layers/dense/MatMul/ReadVariableOp�:full_model/trainable_layers/dense_1/BiasAdd/ReadVariableOp�9full_model/trainable_layers/dense_1/MatMul/ReadVariableOp�?full_model/trainable_layers/output_dense/BiasAdd/ReadVariableOp�>full_model/trainable_layers/output_dense/MatMul/ReadVariableOp�
.full_model/grayscale/rgb_to_grayscale/IdentityIdentitypreprocessing_input*
T0*1
_output_shapes
:������������
1full_model/grayscale/rgb_to_grayscale/Tensordot/bConst*
_output_shapes
:*
dtype0*!
valueB"l	�>�E?�x�=~
4full_model/grayscale/rgb_to_grayscale/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
4full_model/grayscale/rgb_to_grayscale/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          �
5full_model/grayscale/rgb_to_grayscale/Tensordot/ShapeShape7full_model/grayscale/rgb_to_grayscale/Identity:output:0*
T0*
_output_shapes
:
=full_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8full_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2GatherV2>full_model/grayscale/rgb_to_grayscale/Tensordot/Shape:output:0=full_model/grayscale/rgb_to_grayscale/Tensordot/free:output:0Ffull_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
?full_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
:full_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2_1GatherV2>full_model/grayscale/rgb_to_grayscale/Tensordot/Shape:output:0=full_model/grayscale/rgb_to_grayscale/Tensordot/axes:output:0Hfull_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
5full_model/grayscale/rgb_to_grayscale/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
4full_model/grayscale/rgb_to_grayscale/Tensordot/ProdProdAfull_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2:output:0>full_model/grayscale/rgb_to_grayscale/Tensordot/Const:output:0*
T0*
_output_shapes
: �
7full_model/grayscale/rgb_to_grayscale/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
6full_model/grayscale/rgb_to_grayscale/Tensordot/Prod_1ProdCfull_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2_1:output:0@full_model/grayscale/rgb_to_grayscale/Tensordot/Const_1:output:0*
T0*
_output_shapes
: }
;full_model/grayscale/rgb_to_grayscale/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6full_model/grayscale/rgb_to_grayscale/Tensordot/concatConcatV2=full_model/grayscale/rgb_to_grayscale/Tensordot/free:output:0=full_model/grayscale/rgb_to_grayscale/Tensordot/axes:output:0Dfull_model/grayscale/rgb_to_grayscale/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
5full_model/grayscale/rgb_to_grayscale/Tensordot/stackPack=full_model/grayscale/rgb_to_grayscale/Tensordot/Prod:output:0?full_model/grayscale/rgb_to_grayscale/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
9full_model/grayscale/rgb_to_grayscale/Tensordot/transpose	Transpose7full_model/grayscale/rgb_to_grayscale/Identity:output:0?full_model/grayscale/rgb_to_grayscale/Tensordot/concat:output:0*
T0*1
_output_shapes
:������������
7full_model/grayscale/rgb_to_grayscale/Tensordot/ReshapeReshape=full_model/grayscale/rgb_to_grayscale/Tensordot/transpose:y:0>full_model/grayscale/rgb_to_grayscale/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
?full_model/grayscale/rgb_to_grayscale/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9full_model/grayscale/rgb_to_grayscale/Tensordot/Reshape_1Reshape:full_model/grayscale/rgb_to_grayscale/Tensordot/b:output:0Hfull_model/grayscale/rgb_to_grayscale/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
6full_model/grayscale/rgb_to_grayscale/Tensordot/MatMulMatMul@full_model/grayscale/rgb_to_grayscale/Tensordot/Reshape:output:0Bfull_model/grayscale/rgb_to_grayscale/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������z
7full_model/grayscale/rgb_to_grayscale/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
=full_model/grayscale/rgb_to_grayscale/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
8full_model/grayscale/rgb_to_grayscale/Tensordot/concat_1ConcatV2Afull_model/grayscale/rgb_to_grayscale/Tensordot/GatherV2:output:0@full_model/grayscale/rgb_to_grayscale/Tensordot/Const_2:output:0Ffull_model/grayscale/rgb_to_grayscale/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
/full_model/grayscale/rgb_to_grayscale/TensordotReshape@full_model/grayscale/rgb_to_grayscale/Tensordot/MatMul:product:0Afull_model/grayscale/rgb_to_grayscale/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:�����������
4full_model/grayscale/rgb_to_grayscale/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
0full_model/grayscale/rgb_to_grayscale/ExpandDims
ExpandDims8full_model/grayscale/rgb_to_grayscale/Tensordot:output:0=full_model/grayscale/rgb_to_grayscale/ExpandDims/dim:output:0*
T0*1
_output_shapes
:������������
%full_model/grayscale/rgb_to_grayscaleIdentity9full_model/grayscale/rgb_to_grayscale/ExpandDims:output:0*
T0*1
_output_shapes
:������������
'full_model/adaptive_thresholding/Conv2DConv2D.full_model/grayscale/rgb_to_grayscale:output:0.full_model_adaptive_thresholding_conv2d_filter*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
k
&full_model/adaptive_thresholding/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@�
$full_model/adaptive_thresholding/subSub0full_model/adaptive_thresholding/Conv2D:output:0/full_model/adaptive_thresholding/sub/y:output:0*
T0*1
_output_shapes
:������������
*full_model/adaptive_thresholding/LessEqual	LessEqual.full_model/grayscale/rgb_to_grayscale:output:0(full_model/adaptive_thresholding/sub:z:0*
T0*1
_output_shapes
:������������
/full_model/adaptive_thresholding/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
*full_model/adaptive_thresholding/transpose	Transpose.full_model/adaptive_thresholding/LessEqual:z:08full_model/adaptive_thresholding/transpose/perm:output:0*
T0
*1
_output_shapes
:�����������n
+full_model/adaptive_thresholding/SelectV2/tConst*
_output_shapes
: *
dtype0*
value
B :�m
+full_model/adaptive_thresholding/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : �
)full_model/adaptive_thresholding/SelectV2SelectV2.full_model/adaptive_thresholding/transpose:y:04full_model/adaptive_thresholding/SelectV2/t:output:04full_model/adaptive_thresholding/SelectV2/e:output:0*
T0*1
_output_shapes
:������������
1full_model/adaptive_thresholding/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             �
,full_model/adaptive_thresholding/transpose_1	Transpose2full_model/adaptive_thresholding/SelectV2:output:0:full_model/adaptive_thresholding/transpose_1/perm:output:0*
T0*1
_output_shapes
:�����������`
full_model/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;b
full_model/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
full_model/rescaling/Cast_2Cast0full_model/adaptive_thresholding/transpose_1:y:0*

DstT0*

SrcT0*1
_output_shapes
:������������
full_model/rescaling/mulMulfull_model/rescaling/Cast_2:y:0$full_model/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:������������
full_model/rescaling/addAddV2full_model/rescaling/mul:z:0&full_model/rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:������������
8full_model/trainable_layers/conv2d/Conv2D/ReadVariableOpReadVariableOpAfull_model_trainable_layers_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
)full_model/trainable_layers/conv2d/Conv2DConv2Dfull_model/rescaling/add:z:0@full_model/trainable_layers/conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
�
9full_model/trainable_layers/conv2d/BiasAdd/ReadVariableOpReadVariableOpBfull_model_trainable_layers_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*full_model/trainable_layers/conv2d/BiasAddBiasAdd2full_model/trainable_layers/conv2d/Conv2D:output:0Afull_model/trainable_layers/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:�������������
'full_model/trainable_layers/conv2d/ReluRelu3full_model/trainable_layers/conv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:�������������
1full_model/trainable_layers/max_pooling2d/MaxPoolMaxPool5full_model/trainable_layers/conv2d/Relu:activations:0*0
_output_shapes
:���������00�*
ksize
*
paddingVALID*
strides
�
:full_model/trainable_layers/conv2d_1/Conv2D/ReadVariableOpReadVariableOpCfull_model_trainable_layers_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
+full_model/trainable_layers/conv2d_1/Conv2DConv2D:full_model/trainable_layers/max_pooling2d/MaxPool:output:0Bfull_model/trainable_layers/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
�
;full_model/trainable_layers/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpDfull_model_trainable_layers_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
,full_model/trainable_layers/conv2d_1/BiasAddBiasAdd4full_model/trainable_layers/conv2d_1/Conv2D:output:0Cfull_model/trainable_layers/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,��
)full_model/trainable_layers/conv2d_1/ReluRelu5full_model/trainable_layers/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������,,��
3full_model/trainable_layers/max_pooling2d_1/MaxPoolMaxPool7full_model/trainable_layers/conv2d_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
z
)full_model/trainable_layers/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  �
+full_model/trainable_layers/flatten/ReshapeReshape<full_model/trainable_layers/max_pooling2d_1/MaxPool:output:02full_model/trainable_layers/flatten/Const:output:0*
T0*)
_output_shapes
:������������
7full_model/trainable_layers/dense/MatMul/ReadVariableOpReadVariableOp@full_model_trainable_layers_dense_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
(full_model/trainable_layers/dense/MatMulMatMul4full_model/trainable_layers/flatten/Reshape:output:0?full_model/trainable_layers/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
Cfull_model/trainable_layers/batch_normalization/Cast/ReadVariableOpReadVariableOpLfull_model_trainable_layers_batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Efull_model/trainable_layers/batch_normalization/Cast_1/ReadVariableOpReadVariableOpNfull_model_trainable_layers_batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Efull_model/trainable_layers/batch_normalization/Cast_2/ReadVariableOpReadVariableOpNfull_model_trainable_layers_batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Efull_model/trainable_layers/batch_normalization/Cast_3/ReadVariableOpReadVariableOpNfull_model_trainable_layers_batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?full_model/trainable_layers/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
=full_model/trainable_layers/batch_normalization/batchnorm/addAddV2Mfull_model/trainable_layers/batch_normalization/Cast_1/ReadVariableOp:value:0Hfull_model/trainable_layers/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
?full_model/trainable_layers/batch_normalization/batchnorm/RsqrtRsqrtAfull_model/trainable_layers/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
=full_model/trainable_layers/batch_normalization/batchnorm/mulMulCfull_model/trainable_layers/batch_normalization/batchnorm/Rsqrt:y:0Mfull_model/trainable_layers/batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
?full_model/trainable_layers/batch_normalization/batchnorm/mul_1Mul2full_model/trainable_layers/dense/MatMul:product:0Afull_model/trainable_layers/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
?full_model/trainable_layers/batch_normalization/batchnorm/mul_2MulKfull_model/trainable_layers/batch_normalization/Cast/ReadVariableOp:value:0Afull_model/trainable_layers/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
=full_model/trainable_layers/batch_normalization/batchnorm/subSubMfull_model/trainable_layers/batch_normalization/Cast_2/ReadVariableOp:value:0Cfull_model/trainable_layers/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
?full_model/trainable_layers/batch_normalization/batchnorm/add_1AddV2Cfull_model/trainable_layers/batch_normalization/batchnorm/mul_1:z:0Afull_model/trainable_layers/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&full_model/trainable_layers/re_lu/ReluReluCfull_model/trainable_layers/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
9full_model/trainable_layers/dense_1/MatMul/ReadVariableOpReadVariableOpBfull_model_trainable_layers_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
*full_model/trainable_layers/dense_1/MatMulMatMul4full_model/trainable_layers/re_lu/Relu:activations:0Afull_model/trainable_layers/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
:full_model/trainable_layers/dense_1/BiasAdd/ReadVariableOpReadVariableOpCfull_model_trainable_layers_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+full_model/trainable_layers/dense_1/BiasAddBiasAdd4full_model/trainable_layers/dense_1/MatMul:product:0Bfull_model/trainable_layers/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(full_model/trainable_layers/dense_1/ReluRelu4full_model/trainable_layers/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
>full_model/trainable_layers/output_dense/MatMul/ReadVariableOpReadVariableOpGfull_model_trainable_layers_output_dense_matmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0�
/full_model/trainable_layers/output_dense/MatMulMatMul6full_model/trainable_layers/dense_1/Relu:activations:0Ffull_model/trainable_layers/output_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
?full_model/trainable_layers/output_dense/BiasAdd/ReadVariableOpReadVariableOpHfull_model_trainable_layers_output_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
0full_model/trainable_layers/output_dense/BiasAddBiasAdd9full_model/trainable_layers/output_dense/MatMul:product:0Gfull_model/trainable_layers/output_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
2full_model/trainable_layers/softmax_output/SoftmaxSoftmax9full_model/trainable_layers/output_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������1�
IdentityIdentity<full_model/trainable_layers/softmax_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOpD^full_model/trainable_layers/batch_normalization/Cast/ReadVariableOpF^full_model/trainable_layers/batch_normalization/Cast_1/ReadVariableOpF^full_model/trainable_layers/batch_normalization/Cast_2/ReadVariableOpF^full_model/trainable_layers/batch_normalization/Cast_3/ReadVariableOp:^full_model/trainable_layers/conv2d/BiasAdd/ReadVariableOp9^full_model/trainable_layers/conv2d/Conv2D/ReadVariableOp<^full_model/trainable_layers/conv2d_1/BiasAdd/ReadVariableOp;^full_model/trainable_layers/conv2d_1/Conv2D/ReadVariableOp8^full_model/trainable_layers/dense/MatMul/ReadVariableOp;^full_model/trainable_layers/dense_1/BiasAdd/ReadVariableOp:^full_model/trainable_layers/dense_1/MatMul/ReadVariableOp@^full_model/trainable_layers/output_dense/BiasAdd/ReadVariableOp?^full_model/trainable_layers/output_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2�
Cfull_model/trainable_layers/batch_normalization/Cast/ReadVariableOpCfull_model/trainable_layers/batch_normalization/Cast/ReadVariableOp2�
Efull_model/trainable_layers/batch_normalization/Cast_1/ReadVariableOpEfull_model/trainable_layers/batch_normalization/Cast_1/ReadVariableOp2�
Efull_model/trainable_layers/batch_normalization/Cast_2/ReadVariableOpEfull_model/trainable_layers/batch_normalization/Cast_2/ReadVariableOp2�
Efull_model/trainable_layers/batch_normalization/Cast_3/ReadVariableOpEfull_model/trainable_layers/batch_normalization/Cast_3/ReadVariableOp2v
9full_model/trainable_layers/conv2d/BiasAdd/ReadVariableOp9full_model/trainable_layers/conv2d/BiasAdd/ReadVariableOp2t
8full_model/trainable_layers/conv2d/Conv2D/ReadVariableOp8full_model/trainable_layers/conv2d/Conv2D/ReadVariableOp2z
;full_model/trainable_layers/conv2d_1/BiasAdd/ReadVariableOp;full_model/trainable_layers/conv2d_1/BiasAdd/ReadVariableOp2x
:full_model/trainable_layers/conv2d_1/Conv2D/ReadVariableOp:full_model/trainable_layers/conv2d_1/Conv2D/ReadVariableOp2r
7full_model/trainable_layers/dense/MatMul/ReadVariableOp7full_model/trainable_layers/dense/MatMul/ReadVariableOp2x
:full_model/trainable_layers/dense_1/BiasAdd/ReadVariableOp:full_model/trainable_layers/dense_1/BiasAdd/ReadVariableOp2v
9full_model/trainable_layers/dense_1/MatMul/ReadVariableOp9full_model/trainable_layers/dense_1/MatMul/ReadVariableOp2�
?full_model/trainable_layers/output_dense/BiasAdd/ReadVariableOp?full_model/trainable_layers/output_dense/BiasAdd/ReadVariableOp2�
>full_model/trainable_layers/output_dense/MatMul/ReadVariableOp>full_model/trainable_layers/output_dense/MatMul/ReadVariableOp:f b
1
_output_shapes
:�����������
-
_user_specified_namepreprocessing_input:,(
&
_output_shapes
:
�
�
F__inference_full_model_layer_call_and_return_conditional_losses_606428
preprocessing_input 
adaptive_thresholding_6063962
trainable_layers_606400:�&
trainable_layers_606402:	�3
trainable_layers_606404:��&
trainable_layers_606406:	�,
trainable_layers_606408:���&
trainable_layers_606410:	�&
trainable_layers_606412:	�&
trainable_layers_606414:	�&
trainable_layers_606416:	�+
trainable_layers_606418:
��&
trainable_layers_606420:	�*
trainable_layers_606422:	�1%
trainable_layers_606424:1
identity��(trainable_layers/StatefulPartitionedCall�
grayscale/PartitionedCallPartitionedCallpreprocessing_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_grayscale_layer_call_and_return_conditional_losses_606145�
%adaptive_thresholding/PartitionedCallPartitionedCall"grayscale/PartitionedCall:output:0adaptive_thresholding_606396*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606163�
rescaling/PartitionedCallPartitionedCall.adaptive_thresholding/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_rescaling_layer_call_and_return_conditional_losses_606176�
(trainable_layers/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0trainable_layers_606400trainable_layers_606402trainable_layers_606404trainable_layers_606406trainable_layers_606408trainable_layers_606410trainable_layers_606412trainable_layers_606414trainable_layers_606416trainable_layers_606418trainable_layers_606420trainable_layers_606422trainable_layers_606424*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605795�
IdentityIdentity1trainable_layers/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1q
NoOpNoOp)^trainable_layers/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:�����������:: : : : : : : : : : : : : 2T
(trainable_layers/StatefulPartitionedCall(trainable_layers/StatefulPartitionedCall:f b
1
_output_shapes
:�����������
-
_user_specified_namepreprocessing_input:,(
&
_output_shapes
:
�_
�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_607049

inputs@
%conv2d_conv2d_readvariableop_resource:�5
&conv2d_biasadd_readvariableop_resource:	�C
'conv2d_1_conv2d_readvariableop_resource:��7
(conv2d_1_biasadd_readvariableop_resource:	�9
$dense_matmul_readvariableop_resource:���J
;batch_normalization_assignmovingavg_readvariableop_resource:	�L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	�?
0batch_normalization_cast_readvariableop_resource:	�A
2batch_normalization_cast_1_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�>
+output_dense_matmul_readvariableop_resource:	�1:
,output_dense_biasadd_readvariableop_resource:1
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�#output_dense/BiasAdd/ReadVariableOp�"output_dense/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������i
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:�������������
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*0
_output_shapes
:���������00�*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������,,��
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  �
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:������������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
 batch_normalization/moments/meanMeandense/MatMul:product:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
-batch_normalization/moments/SquaredDifferenceSquaredDifferencedense/MatMul:product:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0�
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Muldense/MatMul:product:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
!batch_normalization/batchnorm/subSub/batch_normalization/Cast/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������n

re_lu/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMulre_lu/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
"output_dense/MatMul/ReadVariableOpReadVariableOp+output_dense_matmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0�
output_dense/MatMulMatMuldense_1/Relu:activations:0*output_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
#output_dense/BiasAdd/ReadVariableOpReadVariableOp,output_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
output_dense/BiasAddBiasAddoutput_dense/MatMul:product:0+output_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1r
softmax_output/SoftmaxSoftmaxoutput_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������1o
IdentityIdentity softmax_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp$^output_dense/BiasAdd/ReadVariableOp#^output_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2J
#output_dense/BiasAdd/ReadVariableOp#output_dense/BiasAdd/ReadVariableOp2H
"output_dense/MatMul/ReadVariableOp"output_dense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
1__inference_trainable_layers_layer_call_fn_605824
trainable_input"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
	unknown_3:���
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�1

unknown_11:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalltrainable_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*/
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605795o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nametrainable_input
�
�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_605712

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������,,�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������,,�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������00�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������00�
 
_user_specified_nameinputs
�
�
B__inference_conv2d_layer_call_and_return_conditional_losses_605694

inputs9
conv2d_readvariableop_resource:�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������[
ReluReluBiasAdd:output:0*
T0*2
_output_shapes 
:������������l
IdentityIdentityRelu:activations:0^NoOp*
T0*2
_output_shapes 
:������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_607079

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�$
f
E__inference_grayscale_layer_call_and_return_conditional_losses_606145
input_batch
identityn
rgb_to_grayscale/IdentityIdentityinput_batch*
T0*1
_output_shapes
:�����������q
rgb_to_grayscale/Tensordot/bConst*
_output_shapes
:*
dtype0*!
valueB"l	�>�E?�x�=i
rgb_to_grayscale/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
rgb_to_grayscale/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          r
 rgb_to_grayscale/Tensordot/ShapeShape"rgb_to_grayscale/Identity:output:0*
T0*
_output_shapes
:j
(rgb_to_grayscale/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#rgb_to_grayscale/Tensordot/GatherV2GatherV2)rgb_to_grayscale/Tensordot/Shape:output:0(rgb_to_grayscale/Tensordot/free:output:01rgb_to_grayscale/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*rgb_to_grayscale/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%rgb_to_grayscale/Tensordot/GatherV2_1GatherV2)rgb_to_grayscale/Tensordot/Shape:output:0(rgb_to_grayscale/Tensordot/axes:output:03rgb_to_grayscale/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 rgb_to_grayscale/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
rgb_to_grayscale/Tensordot/ProdProd,rgb_to_grayscale/Tensordot/GatherV2:output:0)rgb_to_grayscale/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"rgb_to_grayscale/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!rgb_to_grayscale/Tensordot/Prod_1Prod.rgb_to_grayscale/Tensordot/GatherV2_1:output:0+rgb_to_grayscale/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&rgb_to_grayscale/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!rgb_to_grayscale/Tensordot/concatConcatV2(rgb_to_grayscale/Tensordot/free:output:0(rgb_to_grayscale/Tensordot/axes:output:0/rgb_to_grayscale/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
 rgb_to_grayscale/Tensordot/stackPack(rgb_to_grayscale/Tensordot/Prod:output:0*rgb_to_grayscale/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
$rgb_to_grayscale/Tensordot/transpose	Transpose"rgb_to_grayscale/Identity:output:0*rgb_to_grayscale/Tensordot/concat:output:0*
T0*1
_output_shapes
:������������
"rgb_to_grayscale/Tensordot/ReshapeReshape(rgb_to_grayscale/Tensordot/transpose:y:0)rgb_to_grayscale/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������{
*rgb_to_grayscale/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
$rgb_to_grayscale/Tensordot/Reshape_1Reshape%rgb_to_grayscale/Tensordot/b:output:03rgb_to_grayscale/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes

:�
!rgb_to_grayscale/Tensordot/MatMulMatMul+rgb_to_grayscale/Tensordot/Reshape:output:0-rgb_to_grayscale/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������e
"rgb_to_grayscale/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB j
(rgb_to_grayscale/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#rgb_to_grayscale/Tensordot/concat_1ConcatV2,rgb_to_grayscale/Tensordot/GatherV2:output:0+rgb_to_grayscale/Tensordot/Const_2:output:01rgb_to_grayscale/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
rgb_to_grayscale/TensordotReshape+rgb_to_grayscale/Tensordot/MatMul:product:0,rgb_to_grayscale/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:�����������j
rgb_to_grayscale/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
rgb_to_grayscale/ExpandDims
ExpandDims#rgb_to_grayscale/Tensordot:output:0(rgb_to_grayscale/ExpandDims/dim:output:0*
T0*1
_output_shapes
:�����������~
rgb_to_grayscaleIdentity$rgb_to_grayscale/ExpandDims:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityrgb_to_grayscale:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_batch
�1
�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606065
trainable_input(
conv2d_606027:�
conv2d_606029:	�+
conv2d_1_606033:��
conv2d_1_606035:	�!
dense_606040:���)
batch_normalization_606043:	�)
batch_normalization_606045:	�)
batch_normalization_606047:	�)
batch_normalization_606049:	�"
dense_1_606053:
��
dense_1_606055:	�&
output_dense_606058:	�1!
output_dense_606060:1
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�$output_dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCalltrainable_inputconv2d_606027conv2d_606029*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_605694�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_605579�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_606033conv2d_1_606035*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������,,�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_605712�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_605591�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_605725�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_606040*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_605734�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_606043batch_normalization_606045batch_normalization_606047batch_normalization_606049*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605618�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_605752�
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_606053dense_1_606055*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_605765�
$output_dense/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_dense_606058output_dense_606060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_output_dense_layer_call_and_return_conditional_losses_605781�
softmax_output/PartitionedCallPartitionedCall-output_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_softmax_output_layer_call_and_return_conditional_losses_605792v
IdentityIdentity'softmax_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall%^output_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$output_dense/StatefulPartitionedCall$output_dense/StatefulPartitionedCall:b ^
1
_output_shapes
:�����������
)
_user_specified_nametrainable_input
�
�
'__inference_conv2d_layer_call_fn_607058

inputs"
unknown:�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_605694z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�E
�

L__inference_trainable_layers_layer_call_and_return_conditional_losses_606979

inputs@
%conv2d_conv2d_readvariableop_resource:�5
&conv2d_biasadd_readvariableop_resource:	�C
'conv2d_1_conv2d_readvariableop_resource:��7
(conv2d_1_biasadd_readvariableop_resource:	�9
$dense_matmul_readvariableop_resource:���?
0batch_normalization_cast_readvariableop_resource:	�A
2batch_normalization_cast_1_readvariableop_resource:	�A
2batch_normalization_cast_2_readvariableop_resource:	�A
2batch_normalization_cast_3_readvariableop_resource:	�:
&dense_1_matmul_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�>
+output_dense_matmul_readvariableop_resource:	�1:
,output_dense_biasadd_readvariableop_resource:1
identity��'batch_normalization/Cast/ReadVariableOp�)batch_normalization/Cast_1/ReadVariableOp�)batch_normalization/Cast_2/ReadVariableOp�)batch_normalization/Cast_3/ReadVariableOp�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�#output_dense/BiasAdd/ReadVariableOp�"output_dense/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:������������i
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*2
_output_shapes 
:�������������
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*0
_output_shapes
:���������00�*
ksize
*
paddingVALID*
strides
�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������,,�k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������,,��
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� y  �
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:������������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'batch_normalization/Cast/ReadVariableOpReadVariableOp0batch_normalization_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization/Cast_1/ReadVariableOpReadVariableOp2batch_normalization_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization/Cast_2/ReadVariableOpReadVariableOp2batch_normalization_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization/Cast_3/ReadVariableOpReadVariableOp2batch_normalization_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV21batch_normalization/Cast_1/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:01batch_normalization/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Muldense/MatMul:product:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
#batch_normalization/batchnorm/mul_2Mul/batch_normalization/Cast/ReadVariableOp:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
!batch_normalization/batchnorm/subSub1batch_normalization/Cast_2/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:����������n

re_lu/ReluRelu'batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_1/MatMulMatMulre_lu/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
"output_dense/MatMul/ReadVariableOpReadVariableOp+output_dense_matmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0�
output_dense/MatMulMatMuldense_1/Relu:activations:0*output_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1�
#output_dense/BiasAdd/ReadVariableOpReadVariableOp,output_dense_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype0�
output_dense/BiasAddBiasAddoutput_dense/MatMul:product:0+output_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1r
softmax_output/SoftmaxSoftmaxoutput_dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������1o
IdentityIdentity softmax_output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp(^batch_normalization/Cast/ReadVariableOp*^batch_normalization/Cast_1/ReadVariableOp*^batch_normalization/Cast_2/ReadVariableOp*^batch_normalization/Cast_3/ReadVariableOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp$^output_dense/BiasAdd/ReadVariableOp#^output_dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 2R
'batch_normalization/Cast/ReadVariableOp'batch_normalization/Cast/ReadVariableOp2V
)batch_normalization/Cast_1/ReadVariableOp)batch_normalization/Cast_1/ReadVariableOp2V
)batch_normalization/Cast_2/ReadVariableOp)batch_normalization/Cast_2/ReadVariableOp2V
)batch_normalization/Cast_3/ReadVariableOp)batch_normalization/Cast_3/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2J
#output_dense/BiasAdd/ReadVariableOp#output_dense/BiasAdd/ReadVariableOp2H
"output_dense/MatMul/ReadVariableOp"output_dense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�1
�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605795

inputs(
conv2d_605695:�
conv2d_605697:	�+
conv2d_1_605713:��
conv2d_1_605715:	�!
dense_605735:���)
batch_normalization_605738:	�)
batch_normalization_605740:	�)
batch_normalization_605742:	�)
batch_normalization_605744:	�"
dense_1_605766:
��
dense_1_605768:	�&
output_dense_605782:	�1!
output_dense_605784:1
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�$output_dense/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_605695conv2d_605697*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_605694�
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������00�* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_605579�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_605713conv2d_1_605715*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������,,�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_605712�
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_605591�
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_605725�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_605735*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_605734�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_605738batch_normalization_605740batch_normalization_605742batch_normalization_605744*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_605618�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_605752�
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0dense_1_605766dense_1_605768*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_605765�
$output_dense/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_dense_605782output_dense_605784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_output_dense_layer_call_and_return_conditional_losses_605781�
softmax_output/PartitionedCallPartitionedCall-output_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_softmax_output_layer_call_and_return_conditional_losses_605792v
IdentityIdentity'softmax_output/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall%^output_dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2L
$output_dense/StatefulPartitionedCall$output_dense/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
1__inference_trainable_layers_layer_call_fn_606923

inputs"
unknown:�
	unknown_0:	�%
	unknown_1:��
	unknown_2:	�
	unknown_3:���
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:
��
	unknown_9:	�

unknown_10:	�1

unknown_11:1
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������1*-
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_trainable_layers_layer_call_and_return_conditional_losses_605964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:�����������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_605765

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
H__inference_output_dense_layer_call_and_return_conditional_losses_605781

inputs1
matmul_readvariableop_resource:	�1-
biasadd_readvariableop_resource:1
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�1*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������1_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������1w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�$
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607214

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�+
cast_readvariableop_resource:	�-
cast_1_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�Cast/ReadVariableOp�Cast_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0m
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:�*
dtype0q
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�n
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�l
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_605734

inputs3
matmul_readvariableop_resource:���
identity��MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityMatMul:product:0^NoOp*
T0*(
_output_shapes
:����������^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:�����������: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
]
preprocessing_inputF
%serving_default_preprocessing_input:0�����������D
trainable_layers0
StatefulPartitionedCall:0���������1tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"layer-0
#layer_with_weights-0
#layer-1
$layer-2
%layer_with_weights-1
%layer-3
&layer-4
'layer-5
(layer_with_weights-2
(layer-6
)layer_with_weights-3
)layer-7
*layer-8
+layer_with_weights-4
+layer-9
,layer_with_weights-5
,layer-10
-layer-11
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_network
~
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12"
trackable_list_wrapper
n
40
51
62
73
84
95
:6
=7
>8
?9
@10"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32�
+__inference_full_model_layer_call_fn_606237
+__inference_full_model_layer_call_fn_606538
+__inference_full_model_layer_call_fn_606571
+__inference_full_model_layer_call_fn_606392�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
�
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32�
F__inference_full_model_layer_call_and_return_conditional_losses_606672
F__inference_full_model_layer_call_and_return_conditional_losses_606787
F__inference_full_model_layer_call_and_return_conditional_losses_606428
F__inference_full_model_layer_call_and_return_conditional_losses_606464�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
�
N	capture_0B�
!__inference__wrapped_model_605570preprocessing_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
	Odecay
Plearning_rate
Qmomentum
Rweight_decay
Siter4momentum�5momentum�6momentum�7momentum�8momentum�9momentum�:momentum�=momentum�>momentum�?momentum�@momentum�"
	optimizer
,
Tserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
*__inference_grayscale_layer_call_fn_606792�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zZtrace_0
�
[trace_02�
E__inference_grayscale_layer_call_and_return_conditional_losses_606824�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z[trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
6__inference_adaptive_thresholding_layer_call_fn_606831�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zatrace_0
�
btrace_02�
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606847�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zbtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
*__inference_rescaling_layer_call_fn_606852�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
�
itrace_02�
E__inference_rescaling_layer_call_and_return_conditional_losses_606861�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
6
j_init_input_shape"
_tf_keras_input_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

4kernel
5bias
 q_jit_compiled_convolution_op"
_tf_keras_layer
�
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

6kernel
7bias
 ~_jit_compiled_convolution_op"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

8kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	9gamma
:beta
;moving_mean
<moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
~
40
51
62
73
84
95
:6
;7
<8
=9
>10
?11
@12"
trackable_list_wrapper
n
40
51
62
73
84
95
:6
=7
>8
?9
@10"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
1__inference_trainable_layers_layer_call_fn_605824
1__inference_trainable_layers_layer_call_fn_606892
1__inference_trainable_layers_layer_call_fn_606923
1__inference_trainable_layers_layer_call_fn_606024�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606979
L__inference_trainable_layers_layer_call_and_return_conditional_losses_607049
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606065
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606106�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
*:(� 2conv2d/kernel
:� 2conv2d/bias
-:+�� 2conv2d_1/kernel
:� 2conv2d_1/bias
#:!��� 2dense/kernel
*:(� 2batch_normalization/gamma
):'� 2batch_normalization/beta
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
$:"
�� 2dense_1/kernel
:� 2dense_1/bias
(:&	�1 2output_dense/kernel
!:1 2output_dense/bias
.
;0
<1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
N	capture_0B�
+__inference_full_model_layer_call_fn_606237preprocessing_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
+__inference_full_model_layer_call_fn_606538inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
+__inference_full_model_layer_call_fn_606571inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
+__inference_full_model_layer_call_fn_606392preprocessing_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
F__inference_full_model_layer_call_and_return_conditional_losses_606672inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
F__inference_full_model_layer_call_and_return_conditional_losses_606787inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
F__inference_full_model_layer_call_and_return_conditional_losses_606428preprocessing_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
F__inference_full_model_layer_call_and_return_conditional_losses_606464preprocessing_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
J
Constjtf.TrackableConstant
:  (2decay
:  (2learning_rate
:  (2momentum
:  (2weight_decay
:	  (2	SGDW/iter
�
N	capture_0B�
$__inference_signature_wrapper_606505preprocessing_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_grayscale_layer_call_fn_606792input_batch"�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_grayscale_layer_call_and_return_conditional_losses_606824input_batch"�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
N	capture_0B�
6__inference_adaptive_thresholding_layer_call_fn_606831input_batch"�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
�
N	capture_0B�
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606847input_batch"�
���
FullArgSpec"
args�
jself
jinput_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zN	capture_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_rescaling_layer_call_fn_606852inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_rescaling_layer_call_and_return_conditional_losses_606861inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_conv2d_layer_call_fn_607058�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_conv2d_layer_call_and_return_conditional_losses_607069�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_max_pooling2d_layer_call_fn_607074�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_607079�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_conv2d_1_layer_call_fn_607088�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_607099�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_1_layer_call_fn_607104�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_607109�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_607114�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_607120�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
'
80"
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_607127�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_607134�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
<
90
:1
;2
<3"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_607147
4__inference_batch_normalization_layer_call_fn_607160�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607180
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607214�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_re_lu_layer_call_fn_607219�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_re_lu_layer_call_and_return_conditional_losses_607224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_607233�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_607244�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_output_dense_layer_call_fn_607253�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_output_dense_layer_call_and_return_conditional_losses_607263�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_softmax_output_layer_call_fn_607268�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_softmax_output_layer_call_and_return_conditional_losses_607273�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
;0
<1"
trackable_list_wrapper
v
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_trainable_layers_layer_call_fn_605824trainable_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_trainable_layers_layer_call_fn_606892inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_trainable_layers_layer_call_fn_606923inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_trainable_layers_layer_call_fn_606024trainable_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606979inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_607049inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606065trainable_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606106trainable_input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
v
�	variables
�	keras_api
�
thresholds
�true_positives
�false_negatives"
_tf_keras_metric
v
�	variables
�	keras_api
�
thresholds
�true_positives
�false_positives"
_tf_keras_metric
�
�	variables
�	keras_api
�
init_shape
�true_positives
�false_positives
�false_negatives
�weights_intermediate"
_tf_keras_metric
�
�	variables
�	keras_api
�true_positives
�true_negatives
�false_positives
�false_negatives"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_conv2d_layer_call_fn_607058inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_conv2d_layer_call_and_return_conditional_losses_607069inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_max_pooling2d_layer_call_fn_607074inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_607079inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_conv2d_1_layer_call_fn_607088inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_607099inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_max_pooling2d_1_layer_call_fn_607104inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_607109inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_flatten_layer_call_fn_607114inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_607120inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dense_layer_call_fn_607127inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_607134inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_layer_call_fn_607147inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_layer_call_fn_607160inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607180inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607214inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_re_lu_layer_call_fn_607219inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_re_lu_layer_call_and_return_conditional_losses_607224inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_1_layer_call_fn_607233inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_607244inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_output_dense_layer_call_fn_607253inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_output_dense_layer_call_and_return_conditional_losses_607263inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_softmax_output_layer_call_fn_607268inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_softmax_output_layer_call_and_return_conditional_losses_607273inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
:1 (2true_positives
:1 (2false_positives
:1 (2false_negatives
$:"1 (2weights_intermediate
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
#:!	�1 (2true_positives
#:!	�1 (2true_negatives
$:"	�1 (2false_positives
$:"	�1 (2false_negatives
6:4� 2SGDW/conv2d/kernel/momentum
(:&� 2SGDW/conv2d/bias/momentum
9:7�� 2SGDW/conv2d_1/kernel/momentum
*:(� 2SGDW/conv2d_1/bias/momentum
/:-��� 2SGDW/dense/kernel/momentum
6:4� 2'SGDW/batch_normalization/gamma/momentum
5:3� 2&SGDW/batch_normalization/beta/momentum
0:.
�� 2SGDW/dense_1/kernel/momentum
):'� 2SGDW/dense_1/bias/momentum
4:2	�1 2!SGDW/output_dense/kernel/momentum
-:+1 2SGDW/output_dense/bias/momentum�
!__inference__wrapped_model_605570�N45678;<:9=>?@F�C
<�9
7�4
preprocessing_input�����������
� "C�@
>
trainable_layers*�'
trainable_layers���������1�
Q__inference_adaptive_thresholding_layer_call_and_return_conditional_losses_606847{N>�;
4�1
/�,
input_batch�����������
� "6�3
,�)
tensor_0�����������
� �
6__inference_adaptive_thresholding_layer_call_fn_606831pN>�;
4�1
/�,
input_batch�����������
� "+�(
unknown������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607180k;<:94�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_607214k;<:94�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
4__inference_batch_normalization_layer_call_fn_607147`;<:94�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
4__inference_batch_normalization_layer_call_fn_607160`;<:94�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
D__inference_conv2d_1_layer_call_and_return_conditional_losses_607099u678�5
.�+
)�&
inputs���������00�
� "5�2
+�(
tensor_0���������,,�
� �
)__inference_conv2d_1_layer_call_fn_607088j678�5
.�+
)�&
inputs���������00�
� "*�'
unknown���������,,��
B__inference_conv2d_layer_call_and_return_conditional_losses_607069x459�6
/�,
*�'
inputs�����������
� "7�4
-�*
tensor_0������������
� �
'__inference_conv2d_layer_call_fn_607058m459�6
/�,
*�'
inputs�����������
� ",�)
unknown�������������
C__inference_dense_1_layer_call_and_return_conditional_losses_607244e=>0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_1_layer_call_fn_607233Z=>0�-
&�#
!�
inputs����������
� ""�
unknown�����������
A__inference_dense_layer_call_and_return_conditional_losses_607134e81�.
'�$
"�
inputs�����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_layer_call_fn_607127Z81�.
'�$
"�
inputs�����������
� ""�
unknown�����������
C__inference_flatten_layer_call_and_return_conditional_losses_607120j8�5
.�+
)�&
inputs����������
� ".�+
$�!
tensor_0�����������
� �
(__inference_flatten_layer_call_fn_607114_8�5
.�+
)�&
inputs����������
� "#� 
unknown������������
F__inference_full_model_layer_call_and_return_conditional_losses_606428�N45678;<:9=>?@N�K
D�A
7�4
preprocessing_input�����������
p 

 
� ",�)
"�
tensor_0���������1
� �
F__inference_full_model_layer_call_and_return_conditional_losses_606464�N45678;<:9=>?@N�K
D�A
7�4
preprocessing_input�����������
p

 
� ",�)
"�
tensor_0���������1
� �
F__inference_full_model_layer_call_and_return_conditional_losses_606672�N45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p 

 
� ",�)
"�
tensor_0���������1
� �
F__inference_full_model_layer_call_and_return_conditional_losses_606787�N45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p

 
� ",�)
"�
tensor_0���������1
� �
+__inference_full_model_layer_call_fn_606237�N45678;<:9=>?@N�K
D�A
7�4
preprocessing_input�����������
p 

 
� "!�
unknown���������1�
+__inference_full_model_layer_call_fn_606392�N45678;<:9=>?@N�K
D�A
7�4
preprocessing_input�����������
p

 
� "!�
unknown���������1�
+__inference_full_model_layer_call_fn_606538vN45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p 

 
� "!�
unknown���������1�
+__inference_full_model_layer_call_fn_606571vN45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p

 
� "!�
unknown���������1�
E__inference_grayscale_layer_call_and_return_conditional_losses_606824x>�;
4�1
/�,
input_batch�����������
� "6�3
,�)
tensor_0�����������
� �
*__inference_grayscale_layer_call_fn_606792m>�;
4�1
/�,
input_batch�����������
� "+�(
unknown������������
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_607109�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_1_layer_call_fn_607104�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_607079�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
.__inference_max_pooling2d_layer_call_fn_607074�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
H__inference_output_dense_layer_call_and_return_conditional_losses_607263d?@0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������1
� �
-__inference_output_dense_layer_call_fn_607253Y?@0�-
&�#
!�
inputs����������
� "!�
unknown���������1�
A__inference_re_lu_layer_call_and_return_conditional_losses_607224a0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
&__inference_re_lu_layer_call_fn_607219V0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_rescaling_layer_call_and_return_conditional_losses_606861s9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
*__inference_rescaling_layer_call_fn_606852h9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
$__inference_signature_wrapper_606505�N45678;<:9=>?@]�Z
� 
S�P
N
preprocessing_input7�4
preprocessing_input�����������"C�@
>
trainable_layers*�'
trainable_layers���������1�
J__inference_softmax_output_layer_call_and_return_conditional_losses_607273_/�,
%�"
 �
inputs���������1
� ",�)
"�
tensor_0���������1
� �
/__inference_softmax_output_layer_call_fn_607268T/�,
%�"
 �
inputs���������1
� "!�
unknown���������1�
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606065�45678;<:9=>?@J�G
@�=
3�0
trainable_input�����������
p 

 
� ",�)
"�
tensor_0���������1
� �
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606106�45678;<:9=>?@J�G
@�=
3�0
trainable_input�����������
p

 
� ",�)
"�
tensor_0���������1
� �
L__inference_trainable_layers_layer_call_and_return_conditional_losses_606979�45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p 

 
� ",�)
"�
tensor_0���������1
� �
L__inference_trainable_layers_layer_call_and_return_conditional_losses_607049�45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p

 
� ",�)
"�
tensor_0���������1
� �
1__inference_trainable_layers_layer_call_fn_605824~45678;<:9=>?@J�G
@�=
3�0
trainable_input�����������
p 

 
� "!�
unknown���������1�
1__inference_trainable_layers_layer_call_fn_606024~45678;<:9=>?@J�G
@�=
3�0
trainable_input�����������
p

 
� "!�
unknown���������1�
1__inference_trainable_layers_layer_call_fn_606892u45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p 

 
� "!�
unknown���������1�
1__inference_trainable_layers_layer_call_fn_606923u45678;<:9=>?@A�>
7�4
*�'
inputs�����������
p

 
� "!�
unknown���������1