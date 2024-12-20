׻
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
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
�
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
�
decoder/conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder/conv2d_32/bias
}
*decoder/conv2d_32/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_32/bias*
_output_shapes
:*
dtype0
�
decoder/conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namedecoder/conv2d_32/kernel
�
,decoder/conv2d_32/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_32/kernel*&
_output_shapes
:@*
dtype0
�
decoder/conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namedecoder/conv2d_31/bias
}
*decoder/conv2d_31/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_31/bias*
_output_shapes
:@*
dtype0
�
decoder/conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*)
shared_namedecoder/conv2d_31/kernel
�
,decoder/conv2d_31/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_31/kernel*'
_output_shapes
:�@*
dtype0
�
>decoder/decoder_block_8/batch_normalization_23/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*O
shared_name@>decoder/decoder_block_8/batch_normalization_23/moving_variance
�
Rdecoder/decoder_block_8/batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp>decoder/decoder_block_8/batch_normalization_23/moving_variance*
_output_shapes	
:�*
dtype0
�
:decoder/decoder_block_8/batch_normalization_23/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:decoder/decoder_block_8/batch_normalization_23/moving_mean
�
Ndecoder/decoder_block_8/batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp:decoder/decoder_block_8/batch_normalization_23/moving_mean*
_output_shapes	
:�*
dtype0
�
3decoder/decoder_block_8/batch_normalization_23/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53decoder/decoder_block_8/batch_normalization_23/beta
�
Gdecoder/decoder_block_8/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_8/batch_normalization_23/beta*
_output_shapes	
:�*
dtype0
�
4decoder/decoder_block_8/batch_normalization_23/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64decoder/decoder_block_8/batch_normalization_23/gamma
�
Hdecoder/decoder_block_8/batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_8/batch_normalization_23/gamma*
_output_shapes	
:�*
dtype0
�
&decoder/decoder_block_8/conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&decoder/decoder_block_8/conv2d_30/bias
�
:decoder/decoder_block_8/conv2d_30/bias/Read/ReadVariableOpReadVariableOp&decoder/decoder_block_8/conv2d_30/bias*
_output_shapes	
:�*
dtype0
�
(decoder/decoder_block_8/conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*9
shared_name*(decoder/decoder_block_8/conv2d_30/kernel
�
<decoder/decoder_block_8/conv2d_30/kernel/Read/ReadVariableOpReadVariableOp(decoder/decoder_block_8/conv2d_30/kernel*(
_output_shapes
:��*
dtype0
�
>decoder/decoder_block_7/batch_normalization_22/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*O
shared_name@>decoder/decoder_block_7/batch_normalization_22/moving_variance
�
Rdecoder/decoder_block_7/batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp>decoder/decoder_block_7/batch_normalization_22/moving_variance*
_output_shapes	
:�*
dtype0
�
:decoder/decoder_block_7/batch_normalization_22/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:decoder/decoder_block_7/batch_normalization_22/moving_mean
�
Ndecoder/decoder_block_7/batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp:decoder/decoder_block_7/batch_normalization_22/moving_mean*
_output_shapes	
:�*
dtype0
�
3decoder/decoder_block_7/batch_normalization_22/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53decoder/decoder_block_7/batch_normalization_22/beta
�
Gdecoder/decoder_block_7/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_7/batch_normalization_22/beta*
_output_shapes	
:�*
dtype0
�
4decoder/decoder_block_7/batch_normalization_22/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64decoder/decoder_block_7/batch_normalization_22/gamma
�
Hdecoder/decoder_block_7/batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_7/batch_normalization_22/gamma*
_output_shapes	
:�*
dtype0
�
&decoder/decoder_block_7/conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&decoder/decoder_block_7/conv2d_29/bias
�
:decoder/decoder_block_7/conv2d_29/bias/Read/ReadVariableOpReadVariableOp&decoder/decoder_block_7/conv2d_29/bias*
_output_shapes	
:�*
dtype0
�
(decoder/decoder_block_7/conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*9
shared_name*(decoder/decoder_block_7/conv2d_29/kernel
�
<decoder/decoder_block_7/conv2d_29/kernel/Read/ReadVariableOpReadVariableOp(decoder/decoder_block_7/conv2d_29/kernel*(
_output_shapes
:��*
dtype0
�
>decoder/decoder_block_6/batch_normalization_21/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*O
shared_name@>decoder/decoder_block_6/batch_normalization_21/moving_variance
�
Rdecoder/decoder_block_6/batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp>decoder/decoder_block_6/batch_normalization_21/moving_variance*
_output_shapes	
:�*
dtype0
�
:decoder/decoder_block_6/batch_normalization_21/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*K
shared_name<:decoder/decoder_block_6/batch_normalization_21/moving_mean
�
Ndecoder/decoder_block_6/batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp:decoder/decoder_block_6/batch_normalization_21/moving_mean*
_output_shapes	
:�*
dtype0
�
3decoder/decoder_block_6/batch_normalization_21/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*D
shared_name53decoder/decoder_block_6/batch_normalization_21/beta
�
Gdecoder/decoder_block_6/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_6/batch_normalization_21/beta*
_output_shapes	
:�*
dtype0
�
4decoder/decoder_block_6/batch_normalization_21/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64decoder/decoder_block_6/batch_normalization_21/gamma
�
Hdecoder/decoder_block_6/batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_6/batch_normalization_21/gamma*
_output_shapes	
:�*
dtype0
�
&decoder/decoder_block_6/conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&decoder/decoder_block_6/conv2d_28/bias
�
:decoder/decoder_block_6/conv2d_28/bias/Read/ReadVariableOpReadVariableOp&decoder/decoder_block_6/conv2d_28/bias*
_output_shapes	
:�*
dtype0
�
(decoder/decoder_block_6/conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*9
shared_name*(decoder/decoder_block_6/conv2d_28/kernel
�
<decoder/decoder_block_6/conv2d_28/kernel/Read/ReadVariableOpReadVariableOp(decoder/decoder_block_6/conv2d_28/kernel*(
_output_shapes
:��*
dtype0
�
decoder/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_namedecoder/dense_8/bias
{
(decoder/dense_8/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_8/bias*
_output_shapes

:��*
dtype0
�
decoder/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_namedecoder/dense_8/kernel
�
*decoder/dense_8/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_8/kernel*!
_output_shapes
:���*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1decoder/dense_8/kerneldecoder/dense_8/bias(decoder/decoder_block_6/conv2d_28/kernel&decoder/decoder_block_6/conv2d_28/bias4decoder/decoder_block_6/batch_normalization_21/gamma3decoder/decoder_block_6/batch_normalization_21/beta:decoder/decoder_block_6/batch_normalization_21/moving_mean>decoder/decoder_block_6/batch_normalization_21/moving_variance(decoder/decoder_block_7/conv2d_29/kernel&decoder/decoder_block_7/conv2d_29/bias4decoder/decoder_block_7/batch_normalization_22/gamma3decoder/decoder_block_7/batch_normalization_22/beta:decoder/decoder_block_7/batch_normalization_22/moving_mean>decoder/decoder_block_7/batch_normalization_22/moving_variance(decoder/decoder_block_8/conv2d_30/kernel&decoder/decoder_block_8/conv2d_30/bias4decoder/decoder_block_8/batch_normalization_23/gamma3decoder/decoder_block_8/batch_normalization_23/beta:decoder/decoder_block_8/batch_normalization_23/moving_mean>decoder/decoder_block_8/batch_normalization_23/moving_variancedecoder/conv2d_31/kerneldecoder/conv2d_31/biasdecoder/conv2d_32/kerneldecoder/conv2d_32/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1923088

NoOpNoOp
�e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�e
value�eB�e B�e
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
firts_layer

	block1


block2

block3
conv
	final

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
#14
$15
%16
&17*
* 
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
,trace_0
-trace_1
.trace_2
/trace_3* 
6
0trace_0
1trace_1
2trace_2
3trace_3* 
* 
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

kernel
bias*
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@conv
	Aupsam
Bbn*
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
	Jupsam
Kbn*
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rconv
	Supsam
Tbn*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

#kernel
$bias
 [_jit_compiled_convolution_op*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

%kernel
&bias
 b_jit_compiled_convolution_op*

cserving_default* 
VP
VARIABLE_VALUEdecoder/dense_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdecoder/dense_8/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(decoder/decoder_block_6/conv2d_28/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&decoder/decoder_block_6/conv2d_28/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4decoder/decoder_block_6/batch_normalization_21/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3decoder/decoder_block_6/batch_normalization_21/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE:decoder/decoder_block_6/batch_normalization_21/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE>decoder/decoder_block_6/batch_normalization_21/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE(decoder/decoder_block_7/conv2d_29/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&decoder/decoder_block_7/conv2d_29/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4decoder/decoder_block_7/batch_normalization_22/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3decoder/decoder_block_7/batch_normalization_22/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:decoder/decoder_block_7/batch_normalization_22/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>decoder/decoder_block_7/batch_normalization_22/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE(decoder/decoder_block_8/conv2d_30/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&decoder/decoder_block_8/conv2d_30/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4decoder/decoder_block_8/batch_normalization_23/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3decoder/decoder_block_8/batch_normalization_23/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE:decoder/decoder_block_8/batch_normalization_23/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE>decoder/decoder_block_8/batch_normalization_23/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_31/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_31/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_32/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_32/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
2
3
!4
"5*
.
0
	1

2
3
4
5*
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
0
1*

0
1*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

itrace_0* 

jtrace_0* 
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

ptrace_0
qtrace_1* 

rtrace_0
strace_1* 
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

kernel
bias
 z_jit_compiled_convolution_op*
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance*
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
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
	�axis
	gamma
beta
moving_mean
moving_variance*
.
0
1
2
 3
!4
"5*
 
0
1
2
 3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
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
	�axis
	gamma
 beta
!moving_mean
"moving_variance*

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

%0
&1*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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
0
1*

@0
A1
B2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
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
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
 
0
1
2
3*

0
1*
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

0
1*

I0
J1
K2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
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
* 
* 
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
 
0
1
2
3*

0
1*
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

!0
"1*

R0
S1
T2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
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
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
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
 
0
 1
!2
"3*

0
 1*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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
0
1*
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
0
1*
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
!0
"1*
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedecoder/dense_8/kerneldecoder/dense_8/bias(decoder/decoder_block_6/conv2d_28/kernel&decoder/decoder_block_6/conv2d_28/bias4decoder/decoder_block_6/batch_normalization_21/gamma3decoder/decoder_block_6/batch_normalization_21/beta:decoder/decoder_block_6/batch_normalization_21/moving_mean>decoder/decoder_block_6/batch_normalization_21/moving_variance(decoder/decoder_block_7/conv2d_29/kernel&decoder/decoder_block_7/conv2d_29/bias4decoder/decoder_block_7/batch_normalization_22/gamma3decoder/decoder_block_7/batch_normalization_22/beta:decoder/decoder_block_7/batch_normalization_22/moving_mean>decoder/decoder_block_7/batch_normalization_22/moving_variance(decoder/decoder_block_8/conv2d_30/kernel&decoder/decoder_block_8/conv2d_30/bias4decoder/decoder_block_8/batch_normalization_23/gamma3decoder/decoder_block_8/batch_normalization_23/beta:decoder/decoder_block_8/batch_normalization_23/moving_mean>decoder/decoder_block_8/batch_normalization_23/moving_variancedecoder/conv2d_31/kerneldecoder/conv2d_31/biasdecoder/conv2d_32/kerneldecoder/conv2d_32/biasConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *)
f$R"
 __inference__traced_save_1924138
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder/dense_8/kerneldecoder/dense_8/bias(decoder/decoder_block_6/conv2d_28/kernel&decoder/decoder_block_6/conv2d_28/bias4decoder/decoder_block_6/batch_normalization_21/gamma3decoder/decoder_block_6/batch_normalization_21/beta:decoder/decoder_block_6/batch_normalization_21/moving_mean>decoder/decoder_block_6/batch_normalization_21/moving_variance(decoder/decoder_block_7/conv2d_29/kernel&decoder/decoder_block_7/conv2d_29/bias4decoder/decoder_block_7/batch_normalization_22/gamma3decoder/decoder_block_7/batch_normalization_22/beta:decoder/decoder_block_7/batch_normalization_22/moving_mean>decoder/decoder_block_7/batch_normalization_22/moving_variance(decoder/decoder_block_8/conv2d_30/kernel&decoder/decoder_block_8/conv2d_30/bias4decoder/decoder_block_8/batch_normalization_23/gamma3decoder/decoder_block_8/batch_normalization_23/beta:decoder/decoder_block_8/batch_normalization_23/moving_mean>decoder/decoder_block_8/batch_normalization_23/moving_variancedecoder/conv2d_31/kerneldecoder/conv2d_31/biasdecoder/conv2d_32/kerneldecoder/conv2d_32/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__traced_restore_1924220��
�	
�
1__inference_decoder_block_7_layer_call_fn_1923544
input_tensor#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922498x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1922103

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_22_layer_call_fn_1923856

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
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1922121�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_23_layer_call_fn_1923922

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1922186�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_1923088
input_1
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�%

unknown_19:�@

unknown_20:@$

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__wrapped_model_1921982y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�*
�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922326
input_tensorD
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�=
.batch_normalization_22_readvariableop_resource:	�?
0batch_normalization_22_readvariableop_1_resource:	�N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_22/AssignNewValue�'batch_normalization_22/AssignNewValue_1�6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_22/ReadVariableOp�'batch_normalization_22/ReadVariableOp_1� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_29/Conv2DConv2Dinput_tensor'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_29/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_22/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:���������  ��
NoOpNoOp&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�	
�
1__inference_decoder_block_6_layer_call_fn_1923452
input_tensor#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922456x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1922411

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�*
�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923665
input_tensorD
(conv2d_30_conv2d_readvariableop_resource:��8
)conv2d_30_biasadd_readvariableop_resource:	�=
.batch_normalization_23_readvariableop_resource:	�?
0batch_normalization_23_readvariableop_1_resource:	�N
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_23/AssignNewValue�'batch_normalization_23/AssignNewValue_1�6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_23/ReadVariableOp�'batch_normalization_23/ReadVariableOp_1� conv2d_30/BiasAdd/ReadVariableOp�conv2d_30/Conv2D/ReadVariableOp�
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_30/Conv2DConv2Dinput_tensor'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_30/Relu:activations:0up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_23/FusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:�������������
NoOpNoOp&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������  �: : : : : : 2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:���������  �
&
_user_specified_nameinput_tensor
�
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923892

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1922121

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923874

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1922394

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�	
�
1__inference_decoder_block_7_layer_call_fn_1923527
input_tensor#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922326x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�

�
D__inference_dense_8_layer_call_and_return_conditional_losses_1923418

inputs3
matmul_readvariableop_resource:���/
biasadd_readvariableop_resource:
��
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������t
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������R
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:�����������c
IdentityIdentityRelu:activations:0^NoOp*
T0*)
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1923194
embedding_input
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�%

unknown_19:�@

unknown_20:@$

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1922741y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1922038

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
D__inference_dense_8_layer_call_and_return_conditional_losses_1922246

inputs3
matmul_readvariableop_resource:���/
biasadd_readvariableop_resource:
��
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������t
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:��*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������R
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:�����������c
IdentityIdentityRelu:activations:0^NoOp*
T0*)
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
M
1__inference_up_sampling2d_8_layer_call_fn_1923897

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1922161�
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
�+
�

D__inference_decoder_layer_call_and_return_conditional_losses_1922418
input_1$
dense_8_1922247:���
dense_8_1922249:
��3
decoder_block_6_1922284:��&
decoder_block_6_1922286:	�&
decoder_block_6_1922288:	�&
decoder_block_6_1922290:	�&
decoder_block_6_1922292:	�&
decoder_block_6_1922294:	�3
decoder_block_7_1922327:��&
decoder_block_7_1922329:	�&
decoder_block_7_1922331:	�&
decoder_block_7_1922333:	�&
decoder_block_7_1922335:	�&
decoder_block_7_1922337:	�3
decoder_block_8_1922370:��&
decoder_block_8_1922372:	�&
decoder_block_8_1922374:	�&
decoder_block_8_1922376:	�&
decoder_block_8_1922378:	�&
decoder_block_8_1922380:	�,
conv2d_31_1922395:�@
conv2d_31_1922397:@+
conv2d_32_1922412:@
conv2d_32_1922414:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�'decoder_block_6/StatefulPartitionedCall�'decoder_block_7/StatefulPartitionedCall�'decoder_block_8/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_8_1922247dense_8_1922249*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1922246f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_8/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
'decoder_block_6/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_6_1922284decoder_block_6_1922286decoder_block_6_1922288decoder_block_6_1922290decoder_block_6_1922292decoder_block_6_1922294*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922283�
'decoder_block_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_6/StatefulPartitionedCall:output:0decoder_block_7_1922327decoder_block_7_1922329decoder_block_7_1922331decoder_block_7_1922333decoder_block_7_1922335decoder_block_7_1922337*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922326�
'decoder_block_8/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_7/StatefulPartitionedCall:output:0decoder_block_8_1922370decoder_block_8_1922372decoder_block_8_1922374decoder_block_8_1922376decoder_block_8_1922378decoder_block_8_1922380*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922369�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_8/StatefulPartitionedCall:output:0conv2d_31_1922395conv2d_31_1922397*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1922394�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_1922412conv2d_32_1922414*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1922411�
IdentityIdentity*conv2d_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall(^decoder_block_6/StatefulPartitionedCall(^decoder_block_7/StatefulPartitionedCall(^decoder_block_8/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2R
'decoder_block_6/StatefulPartitionedCall'decoder_block_6/StatefulPartitionedCall2R
'decoder_block_7/StatefulPartitionedCall'decoder_block_7/StatefulPartitionedCall2R
'decoder_block_8/StatefulPartitionedCall'decoder_block_8/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�,
�

D__inference_decoder_layer_call_and_return_conditional_losses_1922628
embedding_input$
dense_8_1922571:���
dense_8_1922573:
��3
decoder_block_6_1922578:��&
decoder_block_6_1922580:	�&
decoder_block_6_1922582:	�&
decoder_block_6_1922584:	�&
decoder_block_6_1922586:	�&
decoder_block_6_1922588:	�3
decoder_block_7_1922591:��&
decoder_block_7_1922593:	�&
decoder_block_7_1922595:	�&
decoder_block_7_1922597:	�&
decoder_block_7_1922599:	�&
decoder_block_7_1922601:	�3
decoder_block_8_1922604:��&
decoder_block_8_1922606:	�&
decoder_block_8_1922608:	�&
decoder_block_8_1922610:	�&
decoder_block_8_1922612:	�&
decoder_block_8_1922614:	�,
conv2d_31_1922617:�@
conv2d_31_1922619:@+
conv2d_32_1922622:@
conv2d_32_1922624:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�'decoder_block_6/StatefulPartitionedCall�'decoder_block_7/StatefulPartitionedCall�'decoder_block_8/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_8_1922571dense_8_1922573*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1922246f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_8/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
'decoder_block_6/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_6_1922578decoder_block_6_1922580decoder_block_6_1922582decoder_block_6_1922584decoder_block_6_1922586decoder_block_6_1922588*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922283�
'decoder_block_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_6/StatefulPartitionedCall:output:0decoder_block_7_1922591decoder_block_7_1922593decoder_block_7_1922595decoder_block_7_1922597decoder_block_7_1922599decoder_block_7_1922601*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922326�
'decoder_block_8/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_7/StatefulPartitionedCall:output:0decoder_block_8_1922604decoder_block_8_1922606decoder_block_8_1922608decoder_block_8_1922610decoder_block_8_1922612decoder_block_8_1922614*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922369�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_8/StatefulPartitionedCall:output:0conv2d_31_1922617conv2d_31_1922619*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1922394�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_1922622conv2d_32_1922624*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1922411�
IdentityIdentity*conv2d_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall(^decoder_block_6/StatefulPartitionedCall(^decoder_block_7/StatefulPartitionedCall(^decoder_block_8/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2R
'decoder_block_6/StatefulPartitionedCall'decoder_block_6/StatefulPartitionedCall2R
'decoder_block_7/StatefulPartitionedCall'decoder_block_7/StatefulPartitionedCall2R
'decoder_block_8/StatefulPartitionedCall'decoder_block_8/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
M
1__inference_up_sampling2d_6_layer_call_fn_1923739

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1921995�
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
�	
�
1__inference_decoder_block_8_layer_call_fn_1923636
input_tensor#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922540z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������  �: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:���������  �
&
_user_specified_nameinput_tensor
�
�
)__inference_decoder_layer_call_fn_1923141
embedding_input
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�%

unknown_19:�@

unknown_20:@$

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1922628y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�	
�
8__inference_batch_normalization_21_layer_call_fn_1923764

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1922020�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�#
�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923694
input_tensorD
(conv2d_30_conv2d_readvariableop_resource:��8
)conv2d_30_biasadd_readvariableop_resource:	�=
.batch_normalization_23_readvariableop_resource:	�?
0batch_normalization_23_readvariableop_1_resource:	�N
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_23/ReadVariableOp�'batch_normalization_23/ReadVariableOp_1� conv2d_30/BiasAdd/ReadVariableOp�conv2d_30/Conv2D/ReadVariableOp�
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_30/Conv2DConv2Dinput_tensor'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_30/Relu:activations:0up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_23/FusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:�������������
NoOpNoOp7^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������  �: : : : : : 2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:���������  �
&
_user_specified_nameinput_tensor
�
�
+__inference_conv2d_31_layer_call_fn_1923703

inputs"
unknown:�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1922394y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�#
�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923510
input_tensorD
(conv2d_28_conv2d_readvariableop_resource:��8
)conv2d_28_biasadd_readvariableop_resource:	�=
.batch_normalization_21_readvariableop_resource:	�?
0batch_normalization_21_readvariableop_1_resource:	�N
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_21/ReadVariableOp�'batch_normalization_21/ReadVariableOp_1� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_28/Conv2DConv2Dinput_tensor'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_28/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_21/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
�
)__inference_dense_8_layer_call_fn_1923407

inputs
unknown:���
	unknown_0:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1922246q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923795

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�#
�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923602
input_tensorD
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�=
.batch_normalization_22_readvariableop_resource:	�?
0batch_normalization_22_readvariableop_1_resource:	�N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_22/ReadVariableOp�'batch_normalization_22/ReadVariableOp_1� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_29/Conv2DConv2Dinput_tensor'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_29/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_22/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:���������  ��
NoOpNoOp7^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1922186

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�*
�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922283
input_tensorD
(conv2d_28_conv2d_readvariableop_resource:��8
)conv2d_28_biasadd_readvariableop_resource:	�=
.batch_normalization_21_readvariableop_resource:	�?
0batch_normalization_21_readvariableop_1_resource:	�N
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_21/AssignNewValue�'batch_normalization_21/AssignNewValue_1�6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_21/ReadVariableOp�'batch_normalization_21/ReadVariableOp_1� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_28/Conv2DConv2Dinput_tensor'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_28/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_21/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
h
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1923909

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
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
�*
�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923573
input_tensorD
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�=
.batch_normalization_22_readvariableop_resource:	�?
0batch_normalization_22_readvariableop_1_resource:	�N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_22/AssignNewValue�'batch_normalization_22/AssignNewValue_1�6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_22/ReadVariableOp�'batch_normalization_22/ReadVariableOp_1� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_29/Conv2DConv2Dinput_tensor'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_29/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_22/AssignNewValueAssignVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource4batch_normalization_22/FusedBatchNormV3:batch_mean:07^batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_22/AssignNewValue_1AssignVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_22/FusedBatchNormV3:batch_variance:09^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_22/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:���������  ��
NoOpNoOp&^batch_normalization_22/AssignNewValue(^batch_normalization_22/AssignNewValue_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2R
'batch_normalization_22/AssignNewValue_1'batch_normalization_22/AssignNewValue_12N
%batch_normalization_22/AssignNewValue%batch_normalization_22/AssignNewValue2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
��
�
D__inference_decoder_layer_call_and_return_conditional_losses_1923296
embedding_input;
&dense_8_matmul_readvariableop_resource:���7
'dense_8_biasadd_readvariableop_resource:
��T
8decoder_block_6_conv2d_28_conv2d_readvariableop_resource:��H
9decoder_block_6_conv2d_28_biasadd_readvariableop_resource:	�M
>decoder_block_6_batch_normalization_21_readvariableop_resource:	�O
@decoder_block_6_batch_normalization_21_readvariableop_1_resource:	�^
Odecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�`
Qdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�T
8decoder_block_7_conv2d_29_conv2d_readvariableop_resource:��H
9decoder_block_7_conv2d_29_biasadd_readvariableop_resource:	�M
>decoder_block_7_batch_normalization_22_readvariableop_resource:	�O
@decoder_block_7_batch_normalization_22_readvariableop_1_resource:	�^
Odecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�`
Qdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�T
8decoder_block_8_conv2d_30_conv2d_readvariableop_resource:��H
9decoder_block_8_conv2d_30_biasadd_readvariableop_resource:	�M
>decoder_block_8_batch_normalization_23_readvariableop_resource:	�O
@decoder_block_8_batch_normalization_23_readvariableop_1_resource:	�^
Odecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�`
Qdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�C
(conv2d_31_conv2d_readvariableop_resource:�@7
)conv2d_31_biasadd_readvariableop_resource:@B
(conv2d_32_conv2d_readvariableop_resource:@7
)conv2d_32_biasadd_readvariableop_resource:
identity�� conv2d_31/BiasAdd/ReadVariableOp�conv2d_31/Conv2D/ReadVariableOp� conv2d_32/BiasAdd/ReadVariableOp�conv2d_32/Conv2D/ReadVariableOp�5decoder_block_6/batch_normalization_21/AssignNewValue�7decoder_block_6/batch_normalization_21/AssignNewValue_1�Fdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp�Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�5decoder_block_6/batch_normalization_21/ReadVariableOp�7decoder_block_6/batch_normalization_21/ReadVariableOp_1�0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp�/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp�5decoder_block_7/batch_normalization_22/AssignNewValue�7decoder_block_7/batch_normalization_22/AssignNewValue_1�Fdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp�Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�5decoder_block_7/batch_normalization_22/ReadVariableOp�7decoder_block_7/batch_normalization_22/ReadVariableOp_1�0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp�/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp�5decoder_block_8/batch_normalization_23/AssignNewValue�7decoder_block_8/batch_normalization_23/AssignNewValue_1�Fdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp�Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�5decoder_block_8/batch_normalization_23/ReadVariableOp�7decoder_block_8/batch_normalization_23/ReadVariableOp_1�0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp�/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
dense_8/MatMulMatMulembedding_input%dense_8/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������b
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*)
_output_shapes
:�����������f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshapedense_8/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
/decoder_block_6/conv2d_28/Conv2D/ReadVariableOpReadVariableOp8decoder_block_6_conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 decoder_block_6/conv2d_28/Conv2DConv2DReshape:output:07decoder_block_6/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp9decoder_block_6_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!decoder_block_6/conv2d_28/BiasAddBiasAdd)decoder_block_6/conv2d_28/Conv2D:output:08decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
decoder_block_6/conv2d_28/ReluRelu*decoder_block_6/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������v
%decoder_block_6/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'decoder_block_6/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
#decoder_block_6/up_sampling2d_6/mulMul.decoder_block_6/up_sampling2d_6/Const:output:00decoder_block_6/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
<decoder_block_6/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor,decoder_block_6/conv2d_28/Relu:activations:0'decoder_block_6/up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
5decoder_block_6/batch_normalization_21/ReadVariableOpReadVariableOp>decoder_block_6_batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_6/batch_normalization_21/ReadVariableOp_1ReadVariableOp@decoder_block_6_batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpOdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_6/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_6/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0=decoder_block_6/batch_normalization_21/ReadVariableOp:value:0?decoder_block_6/batch_normalization_21/ReadVariableOp_1:value:0Ndecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Pdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5decoder_block_6/batch_normalization_21/AssignNewValueAssignVariableOpOdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceDdecoder_block_6/batch_normalization_21/FusedBatchNormV3:batch_mean:0G^decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7decoder_block_6/batch_normalization_21/AssignNewValue_1AssignVariableOpQdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resourceHdecoder_block_6/batch_normalization_21/FusedBatchNormV3:batch_variance:0I^decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
/decoder_block_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp8decoder_block_7_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 decoder_block_7/conv2d_29/Conv2DConv2D;decoder_block_6/batch_normalization_21/FusedBatchNormV3:y:07decoder_block_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp9decoder_block_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!decoder_block_7/conv2d_29/BiasAddBiasAdd)decoder_block_7/conv2d_29/Conv2D:output:08decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
decoder_block_7/conv2d_29/ReluRelu*decoder_block_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������v
%decoder_block_7/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'decoder_block_7/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
#decoder_block_7/up_sampling2d_7/mulMul.decoder_block_7/up_sampling2d_7/Const:output:00decoder_block_7/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
<decoder_block_7/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor,decoder_block_7/conv2d_29/Relu:activations:0'decoder_block_7/up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
5decoder_block_7/batch_normalization_22/ReadVariableOpReadVariableOp>decoder_block_7_batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_7/batch_normalization_22/ReadVariableOp_1ReadVariableOp@decoder_block_7_batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpOdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_7/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_7/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0=decoder_block_7/batch_normalization_22/ReadVariableOp:value:0?decoder_block_7/batch_normalization_22/ReadVariableOp_1:value:0Ndecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Pdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5decoder_block_7/batch_normalization_22/AssignNewValueAssignVariableOpOdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resourceDdecoder_block_7/batch_normalization_22/FusedBatchNormV3:batch_mean:0G^decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7decoder_block_7/batch_normalization_22/AssignNewValue_1AssignVariableOpQdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resourceHdecoder_block_7/batch_normalization_22/FusedBatchNormV3:batch_variance:0I^decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
/decoder_block_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp8decoder_block_8_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 decoder_block_8/conv2d_30/Conv2DConv2D;decoder_block_7/batch_normalization_22/FusedBatchNormV3:y:07decoder_block_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp9decoder_block_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!decoder_block_8/conv2d_30/BiasAddBiasAdd)decoder_block_8/conv2d_30/Conv2D:output:08decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  ��
decoder_block_8/conv2d_30/ReluRelu*decoder_block_8/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �v
%decoder_block_8/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        x
'decoder_block_8/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
#decoder_block_8/up_sampling2d_8/mulMul.decoder_block_8/up_sampling2d_8/Const:output:00decoder_block_8/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
<decoder_block_8/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor,decoder_block_8/conv2d_30/Relu:activations:0'decoder_block_8/up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
5decoder_block_8/batch_normalization_23/ReadVariableOpReadVariableOp>decoder_block_8_batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_8/batch_normalization_23/ReadVariableOp_1ReadVariableOp@decoder_block_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpOdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_8/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0=decoder_block_8/batch_normalization_23/ReadVariableOp:value:0?decoder_block_8/batch_normalization_23/ReadVariableOp_1:value:0Ndecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Pdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
5decoder_block_8/batch_normalization_23/AssignNewValueAssignVariableOpOdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceDdecoder_block_8/batch_normalization_23/FusedBatchNormV3:batch_mean:0G^decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
7decoder_block_8/batch_normalization_23/AssignNewValue_1AssignVariableOpQdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resourceHdecoder_block_8/batch_normalization_23/FusedBatchNormV3:batch_variance:0I^decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_31/Conv2DConv2D;decoder_block_8/batch_normalization_23/FusedBatchNormV3:y:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@n
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_32/SigmoidSigmoidconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
IdentityIdentityconv2d_32/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp6^decoder_block_6/batch_normalization_21/AssignNewValue8^decoder_block_6/batch_normalization_21/AssignNewValue_1G^decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpI^decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_16^decoder_block_6/batch_normalization_21/ReadVariableOp8^decoder_block_6/batch_normalization_21/ReadVariableOp_11^decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp0^decoder_block_6/conv2d_28/Conv2D/ReadVariableOp6^decoder_block_7/batch_normalization_22/AssignNewValue8^decoder_block_7/batch_normalization_22/AssignNewValue_1G^decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpI^decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_16^decoder_block_7/batch_normalization_22/ReadVariableOp8^decoder_block_7/batch_normalization_22/ReadVariableOp_11^decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp0^decoder_block_7/conv2d_29/Conv2D/ReadVariableOp6^decoder_block_8/batch_normalization_23/AssignNewValue8^decoder_block_8/batch_normalization_23/AssignNewValue_1G^decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpI^decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_16^decoder_block_8/batch_normalization_23/ReadVariableOp8^decoder_block_8/batch_normalization_23/ReadVariableOp_11^decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp0^decoder_block_8/conv2d_30/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2r
7decoder_block_6/batch_normalization_21/AssignNewValue_17decoder_block_6/batch_normalization_21/AssignNewValue_12n
5decoder_block_6/batch_normalization_21/AssignNewValue5decoder_block_6/batch_normalization_21/AssignNewValue2�
Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12�
Fdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpFdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2r
7decoder_block_6/batch_normalization_21/ReadVariableOp_17decoder_block_6/batch_normalization_21/ReadVariableOp_12n
5decoder_block_6/batch_normalization_21/ReadVariableOp5decoder_block_6/batch_normalization_21/ReadVariableOp2d
0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp2b
/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp2r
7decoder_block_7/batch_normalization_22/AssignNewValue_17decoder_block_7/batch_normalization_22/AssignNewValue_12n
5decoder_block_7/batch_normalization_22/AssignNewValue5decoder_block_7/batch_normalization_22/AssignNewValue2�
Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12�
Fdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpFdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2r
7decoder_block_7/batch_normalization_22/ReadVariableOp_17decoder_block_7/batch_normalization_22/ReadVariableOp_12n
5decoder_block_7/batch_normalization_22/ReadVariableOp5decoder_block_7/batch_normalization_22/ReadVariableOp2d
0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp2b
/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp2r
7decoder_block_8/batch_normalization_23/AssignNewValue_17decoder_block_8/batch_normalization_23/AssignNewValue_12n
5decoder_block_8/batch_normalization_23/AssignNewValue5decoder_block_8/batch_normalization_23/AssignNewValue2�
Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12�
Fdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpFdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2r
7decoder_block_8/batch_normalization_23/ReadVariableOp_17decoder_block_8/batch_normalization_23/ReadVariableOp_12n
5decoder_block_8/batch_normalization_23/ReadVariableOp5decoder_block_8/batch_normalization_23/ReadVariableOp2d
0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp2b
/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
�
+__inference_conv2d_32_layer_call_fn_1923723

inputs!
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1922411y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1922078

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
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
�
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923971

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1922020

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�+
�

D__inference_decoder_layer_call_and_return_conditional_losses_1922565
input_1$
dense_8_1922421:���
dense_8_1922423:
��3
decoder_block_6_1922457:��&
decoder_block_6_1922459:	�&
decoder_block_6_1922461:	�&
decoder_block_6_1922463:	�&
decoder_block_6_1922465:	�&
decoder_block_6_1922467:	�3
decoder_block_7_1922499:��&
decoder_block_7_1922501:	�&
decoder_block_7_1922503:	�&
decoder_block_7_1922505:	�&
decoder_block_7_1922507:	�&
decoder_block_7_1922509:	�3
decoder_block_8_1922541:��&
decoder_block_8_1922543:	�&
decoder_block_8_1922545:	�&
decoder_block_8_1922547:	�&
decoder_block_8_1922549:	�&
decoder_block_8_1922551:	�,
conv2d_31_1922554:�@
conv2d_31_1922556:@+
conv2d_32_1922559:@
conv2d_32_1922561:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�'decoder_block_6/StatefulPartitionedCall�'decoder_block_7/StatefulPartitionedCall�'decoder_block_8/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_8_1922421dense_8_1922423*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1922246f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_8/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
'decoder_block_6/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_6_1922457decoder_block_6_1922459decoder_block_6_1922461decoder_block_6_1922463decoder_block_6_1922465decoder_block_6_1922467*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922456�
'decoder_block_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_6/StatefulPartitionedCall:output:0decoder_block_7_1922499decoder_block_7_1922501decoder_block_7_1922503decoder_block_7_1922505decoder_block_7_1922507decoder_block_7_1922509*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922498�
'decoder_block_8/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_7/StatefulPartitionedCall:output:0decoder_block_8_1922541decoder_block_8_1922543decoder_block_8_1922545decoder_block_8_1922547decoder_block_8_1922549decoder_block_8_1922551*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922540�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_8/StatefulPartitionedCall:output:0conv2d_31_1922554conv2d_31_1922556*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1922394�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_1922559conv2d_32_1922561*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1922411�
IdentityIdentity*conv2d_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall(^decoder_block_6/StatefulPartitionedCall(^decoder_block_7/StatefulPartitionedCall(^decoder_block_8/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2R
'decoder_block_6/StatefulPartitionedCall'decoder_block_6/StatefulPartitionedCall2R
'decoder_block_7/StatefulPartitionedCall'decoder_block_7/StatefulPartitionedCall2R
'decoder_block_8/StatefulPartitionedCall'decoder_block_8/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�	
�
8__inference_batch_normalization_23_layer_call_fn_1923935

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1922204�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1923714

inputs9
conv2d_readvariableop_resource:�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:������������
 
_user_specified_nameinputs
�o
�
#__inference__traced_restore_1924220
file_prefix<
'assignvariableop_decoder_dense_8_kernel:���7
'assignvariableop_1_decoder_dense_8_bias:
��W
;assignvariableop_2_decoder_decoder_block_6_conv2d_28_kernel:��H
9assignvariableop_3_decoder_decoder_block_6_conv2d_28_bias:	�V
Gassignvariableop_4_decoder_decoder_block_6_batch_normalization_21_gamma:	�U
Fassignvariableop_5_decoder_decoder_block_6_batch_normalization_21_beta:	�\
Massignvariableop_6_decoder_decoder_block_6_batch_normalization_21_moving_mean:	�`
Qassignvariableop_7_decoder_decoder_block_6_batch_normalization_21_moving_variance:	�W
;assignvariableop_8_decoder_decoder_block_7_conv2d_29_kernel:��H
9assignvariableop_9_decoder_decoder_block_7_conv2d_29_bias:	�W
Hassignvariableop_10_decoder_decoder_block_7_batch_normalization_22_gamma:	�V
Gassignvariableop_11_decoder_decoder_block_7_batch_normalization_22_beta:	�]
Nassignvariableop_12_decoder_decoder_block_7_batch_normalization_22_moving_mean:	�a
Rassignvariableop_13_decoder_decoder_block_7_batch_normalization_22_moving_variance:	�X
<assignvariableop_14_decoder_decoder_block_8_conv2d_30_kernel:��I
:assignvariableop_15_decoder_decoder_block_8_conv2d_30_bias:	�W
Hassignvariableop_16_decoder_decoder_block_8_batch_normalization_23_gamma:	�V
Gassignvariableop_17_decoder_decoder_block_8_batch_normalization_23_beta:	�]
Nassignvariableop_18_decoder_decoder_block_8_batch_normalization_23_moving_mean:	�a
Rassignvariableop_19_decoder_decoder_block_8_batch_normalization_23_moving_variance:	�G
,assignvariableop_20_decoder_conv2d_31_kernel:�@8
*assignvariableop_21_decoder_conv2d_31_bias:@F
,assignvariableop_22_decoder_conv2d_32_kernel:@8
*assignvariableop_23_decoder_conv2d_32_bias:
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp'assignvariableop_decoder_dense_8_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp'assignvariableop_1_decoder_dense_8_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp;assignvariableop_2_decoder_decoder_block_6_conv2d_28_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp9assignvariableop_3_decoder_decoder_block_6_conv2d_28_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpGassignvariableop_4_decoder_decoder_block_6_batch_normalization_21_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpFassignvariableop_5_decoder_decoder_block_6_batch_normalization_21_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpMassignvariableop_6_decoder_decoder_block_6_batch_normalization_21_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpQassignvariableop_7_decoder_decoder_block_6_batch_normalization_21_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp;assignvariableop_8_decoder_decoder_block_7_conv2d_29_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_decoder_decoder_block_7_conv2d_29_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpHassignvariableop_10_decoder_decoder_block_7_batch_normalization_22_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpGassignvariableop_11_decoder_decoder_block_7_batch_normalization_22_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpNassignvariableop_12_decoder_decoder_block_7_batch_normalization_22_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpRassignvariableop_13_decoder_decoder_block_7_batch_normalization_22_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp<assignvariableop_14_decoder_decoder_block_8_conv2d_30_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_decoder_decoder_block_8_conv2d_30_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpHassignvariableop_16_decoder_decoder_block_8_batch_normalization_23_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpGassignvariableop_17_decoder_decoder_block_8_batch_normalization_23_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpNassignvariableop_18_decoder_decoder_block_8_batch_normalization_23_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpRassignvariableop_19_decoder_decoder_block_8_batch_normalization_23_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_decoder_conv2d_31_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_decoder_conv2d_31_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_decoder_conv2d_32_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_decoder_conv2d_32_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
1__inference_decoder_block_8_layer_call_fn_1923619
input_tensor#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922369z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������  �: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:���������  �
&
_user_specified_nameinput_tensor
�#
�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922498
input_tensorD
(conv2d_29_conv2d_readvariableop_resource:��8
)conv2d_29_biasadd_readvariableop_resource:	�=
.batch_normalization_22_readvariableop_resource:	�?
0batch_normalization_22_readvariableop_1_resource:	�N
?batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_22/FusedBatchNormV3/ReadVariableOp�8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_22/ReadVariableOp�'batch_normalization_22/ReadVariableOp_1� conv2d_29/BiasAdd/ReadVariableOp�conv2d_29/Conv2D/ReadVariableOp�
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_29/Conv2DConv2Dinput_tensor'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_29/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_22/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:���������  ��
NoOpNoOp7^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�	
�
8__inference_batch_normalization_22_layer_call_fn_1923843

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
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1922103�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
M
1__inference_up_sampling2d_7_layer_call_fn_1923818

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
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1922078�
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
�
�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1923734

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:�����������d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1922679
input_1
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�%

unknown_19:�@

unknown_20:@$

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1922628y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1922204

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923953

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�#
�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922540
input_tensorD
(conv2d_30_conv2d_readvariableop_resource:��8
)conv2d_30_biasadd_readvariableop_resource:	�=
.batch_normalization_23_readvariableop_resource:	�?
0batch_normalization_23_readvariableop_1_resource:	�N
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_23/ReadVariableOp�'batch_normalization_23/ReadVariableOp_1� conv2d_30/BiasAdd/ReadVariableOp�conv2d_30/Conv2D/ReadVariableOp�
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_30/Conv2DConv2Dinput_tensor'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_30/Relu:activations:0up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_23/FusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:�������������
NoOpNoOp7^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������  �: : : : : : 2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:���������  �
&
_user_specified_nameinput_tensor
��
�
 __inference__traced_save_1924138
file_prefixB
-read_disablecopyonread_decoder_dense_8_kernel:���=
-read_1_disablecopyonread_decoder_dense_8_bias:
��]
Aread_2_disablecopyonread_decoder_decoder_block_6_conv2d_28_kernel:��N
?read_3_disablecopyonread_decoder_decoder_block_6_conv2d_28_bias:	�\
Mread_4_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_gamma:	�[
Lread_5_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_beta:	�b
Sread_6_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_moving_mean:	�f
Wread_7_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_moving_variance:	�]
Aread_8_disablecopyonread_decoder_decoder_block_7_conv2d_29_kernel:��N
?read_9_disablecopyonread_decoder_decoder_block_7_conv2d_29_bias:	�]
Nread_10_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_gamma:	�\
Mread_11_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_beta:	�c
Tread_12_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_moving_mean:	�g
Xread_13_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_moving_variance:	�^
Bread_14_disablecopyonread_decoder_decoder_block_8_conv2d_30_kernel:��O
@read_15_disablecopyonread_decoder_decoder_block_8_conv2d_30_bias:	�]
Nread_16_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_gamma:	�\
Mread_17_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_beta:	�c
Tread_18_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_moving_mean:	�g
Xread_19_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_moving_variance:	�M
2read_20_disablecopyonread_decoder_conv2d_31_kernel:�@>
0read_21_disablecopyonread_decoder_conv2d_31_bias:@L
2read_22_disablecopyonread_decoder_conv2d_32_kernel:@>
0read_23_disablecopyonread_decoder_conv2d_32_bias:
savev2_const
identity_49��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: 
Read/DisableCopyOnReadDisableCopyOnRead-read_disablecopyonread_decoder_dense_8_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp-read_disablecopyonread_decoder_dense_8_kernel^Read/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0l
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���d

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_1/DisableCopyOnReadDisableCopyOnRead-read_1_disablecopyonread_decoder_dense_8_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp-read_1_disablecopyonread_decoder_dense_8_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:��*
dtype0k

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:��a

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:���
Read_2/DisableCopyOnReadDisableCopyOnReadAread_2_disablecopyonread_decoder_decoder_block_6_conv2d_28_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOpAread_2_disablecopyonread_decoder_decoder_block_6_conv2d_28_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0w

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��m

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_3/DisableCopyOnReadDisableCopyOnRead?read_3_disablecopyonread_decoder_decoder_block_6_conv2d_28_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp?read_3_disablecopyonread_decoder_decoder_block_6_conv2d_28_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnReadMread_4_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_gamma"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpMread_4_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_5/DisableCopyOnReadDisableCopyOnReadLread_5_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_beta"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpLread_5_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnReadSread_6_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpSread_6_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_7/DisableCopyOnReadDisableCopyOnReadWread_7_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpWread_7_disablecopyonread_decoder_decoder_block_6_batch_normalization_21_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnReadAread_8_disablecopyonread_decoder_decoder_block_7_conv2d_29_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpAread_8_disablecopyonread_decoder_decoder_block_7_conv2d_29_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0x
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_9/DisableCopyOnReadDisableCopyOnRead?read_9_disablecopyonread_decoder_decoder_block_7_conv2d_29_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp?read_9_disablecopyonread_decoder_decoder_block_7_conv2d_29_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnReadNread_10_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpNread_10_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnReadMread_11_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpMread_11_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnReadTread_12_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpTread_12_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_13/DisableCopyOnReadDisableCopyOnReadXread_13_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpXread_13_disablecopyonread_decoder_decoder_block_7_batch_normalization_22_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnReadBread_14_disablecopyonread_decoder_decoder_block_8_conv2d_30_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpBread_14_disablecopyonread_decoder_decoder_block_8_conv2d_30_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_15/DisableCopyOnReadDisableCopyOnRead@read_15_disablecopyonread_decoder_decoder_block_8_conv2d_30_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp@read_15_disablecopyonread_decoder_decoder_block_8_conv2d_30_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnReadNread_16_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpNread_16_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnReadMread_17_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpMread_17_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnReadTread_18_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpTread_18_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_19/DisableCopyOnReadDisableCopyOnReadXread_19_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpXread_19_disablecopyonread_decoder_decoder_block_8_batch_normalization_23_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_decoder_conv2d_31_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_decoder_conv2d_31_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0x
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_decoder_conv2d_31_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_decoder_conv2d_31_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_decoder_conv2d_32_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_decoder_conv2d_32_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_decoder_conv2d_32_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_decoder_conv2d_32_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
)__inference_decoder_layer_call_fn_1922792
input_1
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�&

unknown_13:��

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:	�

unknown_18:	�%

unknown_19:�@

unknown_20:@$

unknown_21:@

unknown_22:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1922741y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923813

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�,
�

D__inference_decoder_layer_call_and_return_conditional_losses_1922741
embedding_input$
dense_8_1922684:���
dense_8_1922686:
��3
decoder_block_6_1922691:��&
decoder_block_6_1922693:	�&
decoder_block_6_1922695:	�&
decoder_block_6_1922697:	�&
decoder_block_6_1922699:	�&
decoder_block_6_1922701:	�3
decoder_block_7_1922704:��&
decoder_block_7_1922706:	�&
decoder_block_7_1922708:	�&
decoder_block_7_1922710:	�&
decoder_block_7_1922712:	�&
decoder_block_7_1922714:	�3
decoder_block_8_1922717:��&
decoder_block_8_1922719:	�&
decoder_block_8_1922721:	�&
decoder_block_8_1922723:	�&
decoder_block_8_1922725:	�&
decoder_block_8_1922727:	�,
conv2d_31_1922730:�@
conv2d_31_1922732:@+
conv2d_32_1922735:@
conv2d_32_1922737:
identity��!conv2d_31/StatefulPartitionedCall�!conv2d_32/StatefulPartitionedCall�'decoder_block_6/StatefulPartitionedCall�'decoder_block_7/StatefulPartitionedCall�'decoder_block_8/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_8_1922684dense_8_1922686*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1922246f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_8/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
'decoder_block_6/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_6_1922691decoder_block_6_1922693decoder_block_6_1922695decoder_block_6_1922697decoder_block_6_1922699decoder_block_6_1922701*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922456�
'decoder_block_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_6/StatefulPartitionedCall:output:0decoder_block_7_1922704decoder_block_7_1922706decoder_block_7_1922708decoder_block_7_1922710decoder_block_7_1922712decoder_block_7_1922714*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1922498�
'decoder_block_8/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_7/StatefulPartitionedCall:output:0decoder_block_8_1922717decoder_block_8_1922719decoder_block_8_1922721decoder_block_8_1922723decoder_block_8_1922725decoder_block_8_1922727*
Tin
	2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:������������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922540�
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_8/StatefulPartitionedCall:output:0conv2d_31_1922730conv2d_31_1922732*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1922394�
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_1922735conv2d_32_1922737*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1922411�
IdentityIdentity*conv2d_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall(^decoder_block_6/StatefulPartitionedCall(^decoder_block_7/StatefulPartitionedCall(^decoder_block_8/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2R
'decoder_block_6/StatefulPartitionedCall'decoder_block_6/StatefulPartitionedCall2R
'decoder_block_7/StatefulPartitionedCall'decoder_block_7/StatefulPartitionedCall2R
'decoder_block_8/StatefulPartitionedCall'decoder_block_8/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
��
�
"__inference__wrapped_model_1921982
input_1C
.decoder_dense_8_matmul_readvariableop_resource:���?
/decoder_dense_8_biasadd_readvariableop_resource:
��\
@decoder_decoder_block_6_conv2d_28_conv2d_readvariableop_resource:��P
Adecoder_decoder_block_6_conv2d_28_biasadd_readvariableop_resource:	�U
Fdecoder_decoder_block_6_batch_normalization_21_readvariableop_resource:	�W
Hdecoder_decoder_block_6_batch_normalization_21_readvariableop_1_resource:	�f
Wdecoder_decoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�h
Ydecoder_decoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�\
@decoder_decoder_block_7_conv2d_29_conv2d_readvariableop_resource:��P
Adecoder_decoder_block_7_conv2d_29_biasadd_readvariableop_resource:	�U
Fdecoder_decoder_block_7_batch_normalization_22_readvariableop_resource:	�W
Hdecoder_decoder_block_7_batch_normalization_22_readvariableop_1_resource:	�f
Wdecoder_decoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�h
Ydecoder_decoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�\
@decoder_decoder_block_8_conv2d_30_conv2d_readvariableop_resource:��P
Adecoder_decoder_block_8_conv2d_30_biasadd_readvariableop_resource:	�U
Fdecoder_decoder_block_8_batch_normalization_23_readvariableop_resource:	�W
Hdecoder_decoder_block_8_batch_normalization_23_readvariableop_1_resource:	�f
Wdecoder_decoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�h
Ydecoder_decoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�K
0decoder_conv2d_31_conv2d_readvariableop_resource:�@?
1decoder_conv2d_31_biasadd_readvariableop_resource:@J
0decoder_conv2d_32_conv2d_readvariableop_resource:@?
1decoder_conv2d_32_biasadd_readvariableop_resource:
identity��(decoder/conv2d_31/BiasAdd/ReadVariableOp�'decoder/conv2d_31/Conv2D/ReadVariableOp�(decoder/conv2d_32/BiasAdd/ReadVariableOp�'decoder/conv2d_32/Conv2D/ReadVariableOp�Ndecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp�Pdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�=decoder/decoder_block_6/batch_normalization_21/ReadVariableOp�?decoder/decoder_block_6/batch_normalization_21/ReadVariableOp_1�8decoder/decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp�7decoder/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp�Ndecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp�Pdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�=decoder/decoder_block_7/batch_normalization_22/ReadVariableOp�?decoder/decoder_block_7/batch_normalization_22/ReadVariableOp_1�8decoder/decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp�7decoder/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp�Ndecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp�Pdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�=decoder/decoder_block_8/batch_normalization_23/ReadVariableOp�?decoder/decoder_block_8/batch_normalization_23/ReadVariableOp_1�8decoder/decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp�7decoder/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp�&decoder/dense_8/BiasAdd/ReadVariableOp�%decoder/dense_8/MatMul/ReadVariableOp�
%decoder/dense_8/MatMul/ReadVariableOpReadVariableOp.decoder_dense_8_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
decoder/dense_8/MatMulMatMulinput_1-decoder/dense_8/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
&decoder/dense_8/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_8_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
decoder/dense_8/BiasAddBiasAdd decoder/dense_8/MatMul:product:0.decoder/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������r
decoder/dense_8/ReluRelu decoder/dense_8/BiasAdd:output:0*
T0*)
_output_shapes
:�����������n
decoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
decoder/ReshapeReshape"decoder/dense_8/Relu:activations:0decoder/Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
7decoder/decoder_block_6/conv2d_28/Conv2D/ReadVariableOpReadVariableOp@decoder_decoder_block_6_conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
(decoder/decoder_block_6/conv2d_28/Conv2DConv2Ddecoder/Reshape:output:0?decoder/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
8decoder/decoder_block_6/conv2d_28/BiasAdd/ReadVariableOpReadVariableOpAdecoder_decoder_block_6_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)decoder/decoder_block_6/conv2d_28/BiasAddBiasAdd1decoder/decoder_block_6/conv2d_28/Conv2D:output:0@decoder/decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
&decoder/decoder_block_6/conv2d_28/ReluRelu2decoder/decoder_block_6/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������~
-decoder/decoder_block_6/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
/decoder/decoder_block_6/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
+decoder/decoder_block_6/up_sampling2d_6/mulMul6decoder/decoder_block_6/up_sampling2d_6/Const:output:08decoder/decoder_block_6/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
Ddecoder/decoder_block_6/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor4decoder/decoder_block_6/conv2d_28/Relu:activations:0/decoder/decoder_block_6/up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
=decoder/decoder_block_6/batch_normalization_21/ReadVariableOpReadVariableOpFdecoder_decoder_block_6_batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?decoder/decoder_block_6/batch_normalization_21/ReadVariableOp_1ReadVariableOpHdecoder_decoder_block_6_batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Ndecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpWdecoder_decoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Pdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYdecoder_decoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
?decoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3Udecoder/decoder_block_6/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0Edecoder/decoder_block_6/batch_normalization_21/ReadVariableOp:value:0Gdecoder/decoder_block_6/batch_normalization_21/ReadVariableOp_1:value:0Vdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Xdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
7decoder/decoder_block_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp@decoder_decoder_block_7_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
(decoder/decoder_block_7/conv2d_29/Conv2DConv2DCdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3:y:0?decoder/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
8decoder/decoder_block_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOpAdecoder_decoder_block_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)decoder/decoder_block_7/conv2d_29/BiasAddBiasAdd1decoder/decoder_block_7/conv2d_29/Conv2D:output:0@decoder/decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
&decoder/decoder_block_7/conv2d_29/ReluRelu2decoder/decoder_block_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������~
-decoder/decoder_block_7/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
/decoder/decoder_block_7/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
+decoder/decoder_block_7/up_sampling2d_7/mulMul6decoder/decoder_block_7/up_sampling2d_7/Const:output:08decoder/decoder_block_7/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
Ddecoder/decoder_block_7/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor4decoder/decoder_block_7/conv2d_29/Relu:activations:0/decoder/decoder_block_7/up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
=decoder/decoder_block_7/batch_normalization_22/ReadVariableOpReadVariableOpFdecoder_decoder_block_7_batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?decoder/decoder_block_7/batch_normalization_22/ReadVariableOp_1ReadVariableOpHdecoder_decoder_block_7_batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Ndecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpWdecoder_decoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Pdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYdecoder_decoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
?decoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3Udecoder/decoder_block_7/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0Edecoder/decoder_block_7/batch_normalization_22/ReadVariableOp:value:0Gdecoder/decoder_block_7/batch_normalization_22/ReadVariableOp_1:value:0Vdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Xdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
7decoder/decoder_block_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp@decoder_decoder_block_8_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
(decoder/decoder_block_8/conv2d_30/Conv2DConv2DCdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3:y:0?decoder/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
8decoder/decoder_block_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOpAdecoder_decoder_block_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)decoder/decoder_block_8/conv2d_30/BiasAddBiasAdd1decoder/decoder_block_8/conv2d_30/Conv2D:output:0@decoder/decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  ��
&decoder/decoder_block_8/conv2d_30/ReluRelu2decoder/decoder_block_8/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �~
-decoder/decoder_block_8/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        �
/decoder/decoder_block_8/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
+decoder/decoder_block_8/up_sampling2d_8/mulMul6decoder/decoder_block_8/up_sampling2d_8/Const:output:08decoder/decoder_block_8/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
Ddecoder/decoder_block_8/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor4decoder/decoder_block_8/conv2d_30/Relu:activations:0/decoder/decoder_block_8/up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
=decoder/decoder_block_8/batch_normalization_23/ReadVariableOpReadVariableOpFdecoder_decoder_block_8_batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?decoder/decoder_block_8/batch_normalization_23/ReadVariableOp_1ReadVariableOpHdecoder_decoder_block_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Ndecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpWdecoder_decoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Pdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYdecoder_decoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
?decoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3Udecoder/decoder_block_8/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0Edecoder/decoder_block_8/batch_normalization_23/ReadVariableOp:value:0Gdecoder/decoder_block_8/batch_normalization_23/ReadVariableOp_1:value:0Vdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Xdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
is_training( �
'decoder/conv2d_31/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_31_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
decoder/conv2d_31/Conv2DConv2DCdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3:y:0/decoder/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
(decoder/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
decoder/conv2d_31/BiasAddBiasAdd!decoder/conv2d_31/Conv2D:output:00decoder/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@~
decoder/conv2d_31/ReluRelu"decoder/conv2d_31/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
'decoder/conv2d_32/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
decoder/conv2d_32/Conv2DConv2D$decoder/conv2d_31/Relu:activations:0/decoder/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
(decoder/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
decoder/conv2d_32/BiasAddBiasAdd!decoder/conv2d_32/Conv2D:output:00decoder/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
decoder/conv2d_32/SigmoidSigmoid"decoder/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:�����������v
IdentityIdentitydecoder/conv2d_32/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^decoder/conv2d_31/BiasAdd/ReadVariableOp(^decoder/conv2d_31/Conv2D/ReadVariableOp)^decoder/conv2d_32/BiasAdd/ReadVariableOp(^decoder/conv2d_32/Conv2D/ReadVariableOpO^decoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpQ^decoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1>^decoder/decoder_block_6/batch_normalization_21/ReadVariableOp@^decoder/decoder_block_6/batch_normalization_21/ReadVariableOp_19^decoder/decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp8^decoder/decoder_block_6/conv2d_28/Conv2D/ReadVariableOpO^decoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpQ^decoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1>^decoder/decoder_block_7/batch_normalization_22/ReadVariableOp@^decoder/decoder_block_7/batch_normalization_22/ReadVariableOp_19^decoder/decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp8^decoder/decoder_block_7/conv2d_29/Conv2D/ReadVariableOpO^decoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpQ^decoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1>^decoder/decoder_block_8/batch_normalization_23/ReadVariableOp@^decoder/decoder_block_8/batch_normalization_23/ReadVariableOp_19^decoder/decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp8^decoder/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp'^decoder/dense_8/BiasAdd/ReadVariableOp&^decoder/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2T
(decoder/conv2d_31/BiasAdd/ReadVariableOp(decoder/conv2d_31/BiasAdd/ReadVariableOp2R
'decoder/conv2d_31/Conv2D/ReadVariableOp'decoder/conv2d_31/Conv2D/ReadVariableOp2T
(decoder/conv2d_32/BiasAdd/ReadVariableOp(decoder/conv2d_32/BiasAdd/ReadVariableOp2R
'decoder/conv2d_32/Conv2D/ReadVariableOp'decoder/conv2d_32/Conv2D/ReadVariableOp2�
Pdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Pdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12�
Ndecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpNdecoder/decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2�
?decoder/decoder_block_6/batch_normalization_21/ReadVariableOp_1?decoder/decoder_block_6/batch_normalization_21/ReadVariableOp_12~
=decoder/decoder_block_6/batch_normalization_21/ReadVariableOp=decoder/decoder_block_6/batch_normalization_21/ReadVariableOp2t
8decoder/decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp8decoder/decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp2r
7decoder/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp7decoder/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp2�
Pdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Pdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12�
Ndecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpNdecoder/decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2�
?decoder/decoder_block_7/batch_normalization_22/ReadVariableOp_1?decoder/decoder_block_7/batch_normalization_22/ReadVariableOp_12~
=decoder/decoder_block_7/batch_normalization_22/ReadVariableOp=decoder/decoder_block_7/batch_normalization_22/ReadVariableOp2t
8decoder/decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp8decoder/decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp2r
7decoder/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp7decoder/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp2�
Pdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Pdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12�
Ndecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpNdecoder/decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2�
?decoder/decoder_block_8/batch_normalization_23/ReadVariableOp_1?decoder/decoder_block_8/batch_normalization_23/ReadVariableOp_12~
=decoder/decoder_block_8/batch_normalization_23/ReadVariableOp=decoder/decoder_block_8/batch_normalization_23/ReadVariableOp2t
8decoder/decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp8decoder/decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp2r
7decoder/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp7decoder/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp2P
&decoder/dense_8/BiasAdd/ReadVariableOp&decoder/dense_8/BiasAdd/ReadVariableOp2N
%decoder/dense_8/MatMul/ReadVariableOp%decoder/dense_8/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�*
�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1922369
input_tensorD
(conv2d_30_conv2d_readvariableop_resource:��8
)conv2d_30_biasadd_readvariableop_resource:	�=
.batch_normalization_23_readvariableop_resource:	�?
0batch_normalization_23_readvariableop_1_resource:	�N
?batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_23/AssignNewValue�'batch_normalization_23/AssignNewValue_1�6batch_normalization_23/FusedBatchNormV3/ReadVariableOp�8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_23/ReadVariableOp�'batch_normalization_23/ReadVariableOp_1� conv2d_30/BiasAdd/ReadVariableOp�conv2d_30/Conv2D/ReadVariableOp�
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_30/Conv2DConv2Dinput_tensor'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �m
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �f
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_30/Relu:activations:0up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_23/AssignNewValueAssignVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource4batch_normalization_23/FusedBatchNormV3:batch_mean:07^batch_normalization_23/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_23/AssignNewValue_1AssignVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_23/FusedBatchNormV3:batch_variance:09^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_23/FusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:�������������
NoOpNoOp&^batch_normalization_23/AssignNewValue(^batch_normalization_23/AssignNewValue_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:���������  �: : : : : : 2R
'batch_normalization_23/AssignNewValue_1'batch_normalization_23/AssignNewValue_12N
%batch_normalization_23/AssignNewValue%batch_normalization_23/AssignNewValue2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:���������  �
&
_user_specified_nameinput_tensor
��
�
D__inference_decoder_layer_call_and_return_conditional_losses_1923398
embedding_input;
&dense_8_matmul_readvariableop_resource:���7
'dense_8_biasadd_readvariableop_resource:
��T
8decoder_block_6_conv2d_28_conv2d_readvariableop_resource:��H
9decoder_block_6_conv2d_28_biasadd_readvariableop_resource:	�M
>decoder_block_6_batch_normalization_21_readvariableop_resource:	�O
@decoder_block_6_batch_normalization_21_readvariableop_1_resource:	�^
Odecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�`
Qdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�T
8decoder_block_7_conv2d_29_conv2d_readvariableop_resource:��H
9decoder_block_7_conv2d_29_biasadd_readvariableop_resource:	�M
>decoder_block_7_batch_normalization_22_readvariableop_resource:	�O
@decoder_block_7_batch_normalization_22_readvariableop_1_resource:	�^
Odecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resource:	�`
Qdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource:	�T
8decoder_block_8_conv2d_30_conv2d_readvariableop_resource:��H
9decoder_block_8_conv2d_30_biasadd_readvariableop_resource:	�M
>decoder_block_8_batch_normalization_23_readvariableop_resource:	�O
@decoder_block_8_batch_normalization_23_readvariableop_1_resource:	�^
Odecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource:	�`
Qdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource:	�C
(conv2d_31_conv2d_readvariableop_resource:�@7
)conv2d_31_biasadd_readvariableop_resource:@B
(conv2d_32_conv2d_readvariableop_resource:@7
)conv2d_32_biasadd_readvariableop_resource:
identity�� conv2d_31/BiasAdd/ReadVariableOp�conv2d_31/Conv2D/ReadVariableOp� conv2d_32/BiasAdd/ReadVariableOp�conv2d_32/Conv2D/ReadVariableOp�Fdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp�Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�5decoder_block_6/batch_normalization_21/ReadVariableOp�7decoder_block_6/batch_normalization_21/ReadVariableOp_1�0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp�/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp�Fdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp�Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1�5decoder_block_7/batch_normalization_22/ReadVariableOp�7decoder_block_7/batch_normalization_22/ReadVariableOp_1�0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp�/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp�Fdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp�Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1�5decoder_block_8/batch_normalization_23/ReadVariableOp�7decoder_block_8/batch_normalization_23/ReadVariableOp_1�0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp�/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
dense_8/MatMulMatMulembedding_input%dense_8/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������b
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*)
_output_shapes
:�����������f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshapedense_8/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
/decoder_block_6/conv2d_28/Conv2D/ReadVariableOpReadVariableOp8decoder_block_6_conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 decoder_block_6/conv2d_28/Conv2DConv2DReshape:output:07decoder_block_6/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp9decoder_block_6_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!decoder_block_6/conv2d_28/BiasAddBiasAdd)decoder_block_6/conv2d_28/Conv2D:output:08decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
decoder_block_6/conv2d_28/ReluRelu*decoder_block_6/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������v
%decoder_block_6/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'decoder_block_6/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
#decoder_block_6/up_sampling2d_6/mulMul.decoder_block_6/up_sampling2d_6/Const:output:00decoder_block_6/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
<decoder_block_6/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor,decoder_block_6/conv2d_28/Relu:activations:0'decoder_block_6/up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
5decoder_block_6/batch_normalization_21/ReadVariableOpReadVariableOp>decoder_block_6_batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_6/batch_normalization_21/ReadVariableOp_1ReadVariableOp@decoder_block_6_batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpOdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdecoder_block_6_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_6/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_6/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0=decoder_block_6/batch_normalization_21/ReadVariableOp:value:0?decoder_block_6/batch_normalization_21/ReadVariableOp_1:value:0Ndecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Pdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
/decoder_block_7/conv2d_29/Conv2D/ReadVariableOpReadVariableOp8decoder_block_7_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 decoder_block_7/conv2d_29/Conv2DConv2D;decoder_block_6/batch_normalization_21/FusedBatchNormV3:y:07decoder_block_7/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp9decoder_block_7_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!decoder_block_7/conv2d_29/BiasAddBiasAdd)decoder_block_7/conv2d_29/Conv2D:output:08decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
decoder_block_7/conv2d_29/ReluRelu*decoder_block_7/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:����������v
%decoder_block_7/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'decoder_block_7/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
#decoder_block_7/up_sampling2d_7/mulMul.decoder_block_7/up_sampling2d_7/Const:output:00decoder_block_7/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:�
<decoder_block_7/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor,decoder_block_7/conv2d_29/Relu:activations:0'decoder_block_7/up_sampling2d_7/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
5decoder_block_7/batch_normalization_22/ReadVariableOpReadVariableOp>decoder_block_7_batch_normalization_22_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_7/batch_normalization_22/ReadVariableOp_1ReadVariableOp@decoder_block_7_batch_normalization_22_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpOdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdecoder_block_7_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_7/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_7/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0=decoder_block_7/batch_normalization_22/ReadVariableOp:value:0?decoder_block_7/batch_normalization_22/ReadVariableOp_1:value:0Ndecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Pdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
/decoder_block_8/conv2d_30/Conv2D/ReadVariableOpReadVariableOp8decoder_block_8_conv2d_30_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 decoder_block_8/conv2d_30/Conv2DConv2D;decoder_block_7/batch_normalization_22/FusedBatchNormV3:y:07decoder_block_8/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp9decoder_block_8_conv2d_30_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!decoder_block_8/conv2d_30/BiasAddBiasAdd)decoder_block_8/conv2d_30/Conv2D:output:08decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  ��
decoder_block_8/conv2d_30/ReluRelu*decoder_block_8/conv2d_30/BiasAdd:output:0*
T0*0
_output_shapes
:���������  �v
%decoder_block_8/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"        x
'decoder_block_8/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
#decoder_block_8/up_sampling2d_8/mulMul.decoder_block_8/up_sampling2d_8/Const:output:00decoder_block_8/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:�
<decoder_block_8/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor,decoder_block_8/conv2d_30/Relu:activations:0'decoder_block_8/up_sampling2d_8/mul:z:0*
T0*2
_output_shapes 
:������������*
half_pixel_centers(�
5decoder_block_8/batch_normalization_23/ReadVariableOpReadVariableOp>decoder_block_8_batch_normalization_23_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_8/batch_normalization_23/ReadVariableOp_1ReadVariableOp@decoder_block_8_batch_normalization_23_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Fdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpOdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdecoder_block_8_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
7decoder_block_8/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_8/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0=decoder_block_8/batch_normalization_23/ReadVariableOp:value:0?decoder_block_8/batch_normalization_23/ReadVariableOp_1:value:0Ndecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Pdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:������������:�:�:�:�:*
epsilon%o�:*
is_training( �
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_31/Conv2DConv2D;decoder_block_8/batch_normalization_23/FusedBatchNormV3:y:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@n
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_32/SigmoidSigmoidconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
IdentityIdentityconv2d_32/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������

NoOpNoOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOpG^decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpI^decoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_16^decoder_block_6/batch_normalization_21/ReadVariableOp8^decoder_block_6/batch_normalization_21/ReadVariableOp_11^decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp0^decoder_block_6/conv2d_28/Conv2D/ReadVariableOpG^decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpI^decoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_16^decoder_block_7/batch_normalization_22/ReadVariableOp8^decoder_block_7/batch_normalization_22/ReadVariableOp_11^decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp0^decoder_block_7/conv2d_29/Conv2D/ReadVariableOpG^decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpI^decoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_16^decoder_block_8/batch_normalization_23/ReadVariableOp8^decoder_block_8/batch_normalization_23/ReadVariableOp_11^decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp0^decoder_block_8/conv2d_30/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2�
Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Hdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12�
Fdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOpFdecoder_block_6/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2r
7decoder_block_6/batch_normalization_21/ReadVariableOp_17decoder_block_6/batch_normalization_21/ReadVariableOp_12n
5decoder_block_6/batch_normalization_21/ReadVariableOp5decoder_block_6/batch_normalization_21/ReadVariableOp2d
0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp0decoder_block_6/conv2d_28/BiasAdd/ReadVariableOp2b
/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp/decoder_block_6/conv2d_28/Conv2D/ReadVariableOp2�
Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Hdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12�
Fdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOpFdecoder_block_7/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2r
7decoder_block_7/batch_normalization_22/ReadVariableOp_17decoder_block_7/batch_normalization_22/ReadVariableOp_12n
5decoder_block_7/batch_normalization_22/ReadVariableOp5decoder_block_7/batch_normalization_22/ReadVariableOp2d
0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp0decoder_block_7/conv2d_29/BiasAdd/ReadVariableOp2b
/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp/decoder_block_7/conv2d_29/Conv2D/ReadVariableOp2�
Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Hdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12�
Fdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOpFdecoder_block_8/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2r
7decoder_block_8/batch_normalization_23/ReadVariableOp_17decoder_block_8/batch_normalization_23/ReadVariableOp_12n
5decoder_block_8/batch_normalization_23/ReadVariableOp5decoder_block_8/batch_normalization_23/ReadVariableOp2d
0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp0decoder_block_8/conv2d_30/BiasAdd/ReadVariableOp2b
/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp/decoder_block_8/conv2d_30/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�	
�
1__inference_decoder_block_6_layer_call_fn_1923435
input_tensor#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922283x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
h
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1923830

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
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
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1921995

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
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
�
8__inference_batch_normalization_21_layer_call_fn_1923777

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1922038�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
h
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1922161

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
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
�#
�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1922456
input_tensorD
(conv2d_28_conv2d_readvariableop_resource:��8
)conv2d_28_biasadd_readvariableop_resource:	�=
.batch_normalization_21_readvariableop_resource:	�?
0batch_normalization_21_readvariableop_1_resource:	�N
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�
identity��6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_21/ReadVariableOp�'batch_normalization_21/ReadVariableOp_1� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_28/Conv2DConv2Dinput_tensor'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_28/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_21/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp7^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�*
�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923481
input_tensorD
(conv2d_28_conv2d_readvariableop_resource:��8
)conv2d_28_biasadd_readvariableop_resource:	�=
.batch_normalization_21_readvariableop_resource:	�?
0batch_normalization_21_readvariableop_1_resource:	�N
?batch_normalization_21_fusedbatchnormv3_readvariableop_resource:	�P
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource:	�
identity��%batch_normalization_21/AssignNewValue�'batch_normalization_21/AssignNewValue_1�6batch_normalization_21/FusedBatchNormV3/ReadVariableOp�8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_21/ReadVariableOp�'batch_normalization_21/ReadVariableOp_1� conv2d_28/BiasAdd/ReadVariableOp�conv2d_28/Conv2D/ReadVariableOp�
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_28/Conv2DConv2Dinput_tensor'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������m
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:����������f
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_28/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_21/AssignNewValueAssignVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource4batch_normalization_21/FusedBatchNormV3:batch_mean:07^batch_normalization_21/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_21/AssignNewValue_1AssignVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_21/FusedBatchNormV3:batch_variance:09^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_21/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp&^batch_normalization_21/AssignNewValue(^batch_normalization_21/AssignNewValue_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2R
'batch_normalization_21/AssignNewValue_1'batch_normalization_21/AssignNewValue_12N
%batch_normalization_21/AssignNewValue%batch_normalization_21/AssignNewValue2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:����������
&
_user_specified_nameinput_tensor
�
h
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1923751

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
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
<
input_11
serving_default_input_1:0����������F
output_1:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
firts_layer

	block1


block2

block3
conv
	final

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
#14
$15
%16
&17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
,trace_0
-trace_1
.trace_2
/trace_32�
)__inference_decoder_layer_call_fn_1922679
)__inference_decoder_layer_call_fn_1922792
)__inference_decoder_layer_call_fn_1923141
)__inference_decoder_layer_call_fn_1923194�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z,trace_0z-trace_1z.trace_2z/trace_3
�
0trace_0
1trace_1
2trace_2
3trace_32�
D__inference_decoder_layer_call_and_return_conditional_losses_1922418
D__inference_decoder_layer_call_and_return_conditional_losses_1922565
D__inference_decoder_layer_call_and_return_conditional_losses_1923296
D__inference_decoder_layer_call_and_return_conditional_losses_1923398�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z0trace_0z1trace_1z2trace_2z3trace_3
�B�
"__inference__wrapped_model_1921982input_1"�
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
 
�
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@conv
	Aupsam
Bbn"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
	Jupsam
Kbn"
_tf_keras_layer
�
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rconv
	Supsam
Tbn"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

#kernel
$bias
 [_jit_compiled_convolution_op"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

%kernel
&bias
 b_jit_compiled_convolution_op"
_tf_keras_layer
,
cserving_default"
signature_map
+:)���2decoder/dense_8/kernel
$:"��2decoder/dense_8/bias
D:B��2(decoder/decoder_block_6/conv2d_28/kernel
5:3�2&decoder/decoder_block_6/conv2d_28/bias
C:A�24decoder/decoder_block_6/batch_normalization_21/gamma
B:@�23decoder/decoder_block_6/batch_normalization_21/beta
K:I� (2:decoder/decoder_block_6/batch_normalization_21/moving_mean
O:M� (2>decoder/decoder_block_6/batch_normalization_21/moving_variance
D:B��2(decoder/decoder_block_7/conv2d_29/kernel
5:3�2&decoder/decoder_block_7/conv2d_29/bias
C:A�24decoder/decoder_block_7/batch_normalization_22/gamma
B:@�23decoder/decoder_block_7/batch_normalization_22/beta
K:I� (2:decoder/decoder_block_7/batch_normalization_22/moving_mean
O:M� (2>decoder/decoder_block_7/batch_normalization_22/moving_variance
D:B��2(decoder/decoder_block_8/conv2d_30/kernel
5:3�2&decoder/decoder_block_8/conv2d_30/bias
C:A�24decoder/decoder_block_8/batch_normalization_23/gamma
B:@�23decoder/decoder_block_8/batch_normalization_23/beta
K:I� (2:decoder/decoder_block_8/batch_normalization_23/moving_mean
O:M� (2>decoder/decoder_block_8/batch_normalization_23/moving_variance
3:1�@2decoder/conv2d_31/kernel
$:"@2decoder/conv2d_31/bias
2:0@2decoder/conv2d_32/kernel
$:"2decoder/conv2d_32/bias
J
0
1
2
3
!4
"5"
trackable_list_wrapper
J
0
	1

2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decoder_layer_call_fn_1922679input_1"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
)__inference_decoder_layer_call_fn_1922792input_1"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
)__inference_decoder_layer_call_fn_1923141embedding_input"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
)__inference_decoder_layer_call_fn_1923194embedding_input"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
D__inference_decoder_layer_call_and_return_conditional_losses_1922418input_1"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
D__inference_decoder_layer_call_and_return_conditional_losses_1922565input_1"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
D__inference_decoder_layer_call_and_return_conditional_losses_1923296embedding_input"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
D__inference_decoder_layer_call_and_return_conditional_losses_1923398embedding_input"�
���
FullArgSpec
args�
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
�
itrace_02�
)__inference_dense_8_layer_call_fn_1923407�
���
FullArgSpec
args�

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
annotations� *
 zitrace_0
�
jtrace_02�
D__inference_dense_8_layer_call_and_return_conditional_losses_1923418�
���
FullArgSpec
args�

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
annotations� *
 zjtrace_0
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
ptrace_0
qtrace_12�
1__inference_decoder_block_6_layer_call_fn_1923435
1__inference_decoder_block_6_layer_call_fn_1923452�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 zptrace_0zqtrace_1
�
rtrace_0
strace_12�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923481
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923510�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 zrtrace_0zstrace_1
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

kernel
bias
 z_jit_compiled_convolution_op"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_decoder_block_7_layer_call_fn_1923527
1__inference_decoder_block_7_layer_call_fn_1923544�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923573
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923602�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
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
	�axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
1
2
 3
!4
"5"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_decoder_block_8_layer_call_fn_1923619
1__inference_decoder_block_8_layer_call_fn_1923636�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923665
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923694�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
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
	�axis
	gamma
 beta
!moving_mean
"moving_variance"
_tf_keras_layer
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_31_layer_call_fn_1923703�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1923714�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_conv2d_32_layer_call_fn_1923723�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1923734�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
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
annotations� *
 0
�B�
%__inference_signature_wrapper_1923088input_1"�
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
)__inference_dense_8_layer_call_fn_1923407inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_8_layer_call_and_return_conditional_losses_1923418inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
0
1"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_decoder_block_6_layer_call_fn_1923435input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
1__inference_decoder_block_6_layer_call_fn_1923452input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923481input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923510input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�
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
annotations� *
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
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_6_layer_call_fn_1923739�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1923751�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
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
8__inference_batch_normalization_21_layer_call_fn_1923764
8__inference_batch_normalization_21_layer_call_fn_1923777�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923795
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923813�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_decoder_block_7_layer_call_fn_1923527input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
1__inference_decoder_block_7_layer_call_fn_1923544input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923573input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923602input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
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
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�
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
annotations� *
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
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_7_layer_call_fn_1923818�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1923830�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
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
8__inference_batch_normalization_22_layer_call_fn_1923843
8__inference_batch_normalization_22_layer_call_fn_1923856�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923874
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923892�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
5
R0
S1
T2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_decoder_block_8_layer_call_fn_1923619input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
1__inference_decoder_block_8_layer_call_fn_1923636input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923665input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
�B�
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923694input_tensor"�
���
FullArgSpec'
args�
jinput_tensor

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
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
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
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�

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
annotations� *
 
�2��
���
FullArgSpec
args�
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
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
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
1__inference_up_sampling2d_8_layer_call_fn_1923897�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1923909�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
<
0
 1
!2
"3"
trackable_list_wrapper
.
0
 1"
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
�
�trace_0
�trace_12�
8__inference_batch_normalization_23_layer_call_fn_1923922
8__inference_batch_normalization_23_layer_call_fn_1923935�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923953
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923971�
���
FullArgSpec)
args!�
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
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
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
+__inference_conv2d_31_layer_call_fn_1923703inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1923714inputs"�
���
FullArgSpec
args�

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
annotations� *
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
+__inference_conv2d_32_layer_call_fn_1923723inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1923734inputs"�
���
FullArgSpec
args�

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
annotations� *
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
1__inference_up_sampling2d_6_layer_call_fn_1923739inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1923751inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
0
1"
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
8__inference_batch_normalization_21_layer_call_fn_1923764inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
8__inference_batch_normalization_21_layer_call_fn_1923777inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923795inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923813inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
1__inference_up_sampling2d_7_layer_call_fn_1923818inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1923830inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
0
1"
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
8__inference_batch_normalization_22_layer_call_fn_1923843inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
8__inference_batch_normalization_22_layer_call_fn_1923856inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923874inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923892inputs"�
���
FullArgSpec)
args!�
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
annotations� *
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
1__inference_up_sampling2d_8_layer_call_fn_1923897inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1923909inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
!0
"1"
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
8__inference_batch_normalization_23_layer_call_fn_1923922inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
8__inference_batch_normalization_23_layer_call_fn_1923935inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923953inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 
�B�
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923971inputs"�
���
FullArgSpec)
args!�
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
annotations� *
 �
"__inference__wrapped_model_1921982� !"#$%&1�.
'�$
"�
input_1����������
� "=�:
8
output_1,�)
output_1������������
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923795�R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
S__inference_batch_normalization_21_layer_call_and_return_conditional_losses_1923813�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
8__inference_batch_normalization_21_layer_call_fn_1923764�R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
8__inference_batch_normalization_21_layer_call_fn_1923777�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923874�R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
S__inference_batch_normalization_22_layer_call_and_return_conditional_losses_1923892�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
8__inference_batch_normalization_22_layer_call_fn_1923843�R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
8__inference_batch_normalization_22_layer_call_fn_1923856�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923953� !"R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
S__inference_batch_normalization_23_layer_call_and_return_conditional_losses_1923971� !"R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
8__inference_batch_normalization_23_layer_call_fn_1923922� !"R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
8__inference_batch_normalization_23_layer_call_fn_1923935� !"R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
F__inference_conv2d_31_layer_call_and_return_conditional_losses_1923714x#$:�7
0�-
+�(
inputs������������
� "6�3
,�)
tensor_0�����������@
� �
+__inference_conv2d_31_layer_call_fn_1923703m#$:�7
0�-
+�(
inputs������������
� "+�(
unknown�����������@�
F__inference_conv2d_32_layer_call_and_return_conditional_losses_1923734w%&9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������
� �
+__inference_conv2d_32_layer_call_fn_1923723l%&9�6
/�,
*�'
inputs�����������@
� "+�(
unknown������������
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923481�B�?
8�5
/�,
input_tensor����������
p
� "5�2
+�(
tensor_0����������
� �
L__inference_decoder_block_6_layer_call_and_return_conditional_losses_1923510�B�?
8�5
/�,
input_tensor����������
p 
� "5�2
+�(
tensor_0����������
� �
1__inference_decoder_block_6_layer_call_fn_1923435xB�?
8�5
/�,
input_tensor����������
p
� "*�'
unknown�����������
1__inference_decoder_block_6_layer_call_fn_1923452xB�?
8�5
/�,
input_tensor����������
p 
� "*�'
unknown�����������
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923573�B�?
8�5
/�,
input_tensor����������
p
� "5�2
+�(
tensor_0���������  �
� �
L__inference_decoder_block_7_layer_call_and_return_conditional_losses_1923602�B�?
8�5
/�,
input_tensor����������
p 
� "5�2
+�(
tensor_0���������  �
� �
1__inference_decoder_block_7_layer_call_fn_1923527xB�?
8�5
/�,
input_tensor����������
p
� "*�'
unknown���������  ��
1__inference_decoder_block_7_layer_call_fn_1923544xB�?
8�5
/�,
input_tensor����������
p 
� "*�'
unknown���������  ��
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923665� !"B�?
8�5
/�,
input_tensor���������  �
p
� "7�4
-�*
tensor_0������������
� �
L__inference_decoder_block_8_layer_call_and_return_conditional_losses_1923694� !"B�?
8�5
/�,
input_tensor���������  �
p 
� "7�4
-�*
tensor_0������������
� �
1__inference_decoder_block_8_layer_call_fn_1923619z !"B�?
8�5
/�,
input_tensor���������  �
p
� ",�)
unknown�������������
1__inference_decoder_block_8_layer_call_fn_1923636z !"B�?
8�5
/�,
input_tensor���������  �
p 
� ",�)
unknown�������������
D__inference_decoder_layer_call_and_return_conditional_losses_1922418� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1922565� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1923296� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1923398� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
)__inference_decoder_layer_call_fn_1922679� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp"+�(
unknown������������
)__inference_decoder_layer_call_fn_1922792� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp "+�(
unknown������������
)__inference_decoder_layer_call_fn_1923141� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp"+�(
unknown������������
)__inference_decoder_layer_call_fn_1923194� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp "+�(
unknown������������
D__inference_dense_8_layer_call_and_return_conditional_losses_1923418f0�-
&�#
!�
inputs����������
� ".�+
$�!
tensor_0�����������
� �
)__inference_dense_8_layer_call_fn_1923407[0�-
&�#
!�
inputs����������
� "#� 
unknown������������
%__inference_signature_wrapper_1923088� !"#$%&<�9
� 
2�/
-
input_1"�
input_1����������"=�:
8
output_1,�)
output_1������������
L__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_1923751�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_6_layer_call_fn_1923739�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_1923830�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_7_layer_call_fn_1923818�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1923909�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_8_layer_call_fn_1923897�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4������������������������������������