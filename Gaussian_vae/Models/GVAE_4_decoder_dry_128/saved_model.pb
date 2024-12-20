�
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
decoder/conv2d_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder/conv2d_76/bias
}
*decoder/conv2d_76/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_76/bias*
_output_shapes
:*
dtype0
�
decoder/conv2d_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namedecoder/conv2d_76/kernel
�
,decoder/conv2d_76/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_76/kernel*&
_output_shapes
:*
dtype0
�
decoder/conv2d_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder/conv2d_75/bias
}
*decoder/conv2d_75/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_75/bias*
_output_shapes
:*
dtype0
�
decoder/conv2d_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namedecoder/conv2d_75/kernel
�
,decoder/conv2d_75/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_75/kernel*&
_output_shapes
:*
dtype0
�
?decoder/decoder_block_20/batch_normalization_55/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?decoder/decoder_block_20/batch_normalization_55/moving_variance
�
Sdecoder/decoder_block_20/batch_normalization_55/moving_variance/Read/ReadVariableOpReadVariableOp?decoder/decoder_block_20/batch_normalization_55/moving_variance*
_output_shapes
:*
dtype0
�
;decoder/decoder_block_20/batch_normalization_55/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;decoder/decoder_block_20/batch_normalization_55/moving_mean
�
Odecoder/decoder_block_20/batch_normalization_55/moving_mean/Read/ReadVariableOpReadVariableOp;decoder/decoder_block_20/batch_normalization_55/moving_mean*
_output_shapes
:*
dtype0
�
4decoder/decoder_block_20/batch_normalization_55/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64decoder/decoder_block_20/batch_normalization_55/beta
�
Hdecoder/decoder_block_20/batch_normalization_55/beta/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_20/batch_normalization_55/beta*
_output_shapes
:*
dtype0
�
5decoder/decoder_block_20/batch_normalization_55/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75decoder/decoder_block_20/batch_normalization_55/gamma
�
Idecoder/decoder_block_20/batch_normalization_55/gamma/Read/ReadVariableOpReadVariableOp5decoder/decoder_block_20/batch_normalization_55/gamma*
_output_shapes
:*
dtype0
�
'decoder/decoder_block_20/conv2d_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'decoder/decoder_block_20/conv2d_74/bias
�
;decoder/decoder_block_20/conv2d_74/bias/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_20/conv2d_74/bias*
_output_shapes
:*
dtype0
�
)decoder/decoder_block_20/conv2d_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)decoder/decoder_block_20/conv2d_74/kernel
�
=decoder/decoder_block_20/conv2d_74/kernel/Read/ReadVariableOpReadVariableOp)decoder/decoder_block_20/conv2d_74/kernel*&
_output_shapes
: *
dtype0
�
?decoder/decoder_block_19/batch_normalization_54/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *P
shared_nameA?decoder/decoder_block_19/batch_normalization_54/moving_variance
�
Sdecoder/decoder_block_19/batch_normalization_54/moving_variance/Read/ReadVariableOpReadVariableOp?decoder/decoder_block_19/batch_normalization_54/moving_variance*
_output_shapes
: *
dtype0
�
;decoder/decoder_block_19/batch_normalization_54/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;decoder/decoder_block_19/batch_normalization_54/moving_mean
�
Odecoder/decoder_block_19/batch_normalization_54/moving_mean/Read/ReadVariableOpReadVariableOp;decoder/decoder_block_19/batch_normalization_54/moving_mean*
_output_shapes
: *
dtype0
�
4decoder/decoder_block_19/batch_normalization_54/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64decoder/decoder_block_19/batch_normalization_54/beta
�
Hdecoder/decoder_block_19/batch_normalization_54/beta/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_19/batch_normalization_54/beta*
_output_shapes
: *
dtype0
�
5decoder/decoder_block_19/batch_normalization_54/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75decoder/decoder_block_19/batch_normalization_54/gamma
�
Idecoder/decoder_block_19/batch_normalization_54/gamma/Read/ReadVariableOpReadVariableOp5decoder/decoder_block_19/batch_normalization_54/gamma*
_output_shapes
: *
dtype0
�
'decoder/decoder_block_19/conv2d_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'decoder/decoder_block_19/conv2d_73/bias
�
;decoder/decoder_block_19/conv2d_73/bias/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_19/conv2d_73/bias*
_output_shapes
: *
dtype0
�
)decoder/decoder_block_19/conv2d_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *:
shared_name+)decoder/decoder_block_19/conv2d_73/kernel
�
=decoder/decoder_block_19/conv2d_73/kernel/Read/ReadVariableOpReadVariableOp)decoder/decoder_block_19/conv2d_73/kernel*&
_output_shapes
:@ *
dtype0
�
?decoder/decoder_block_18/batch_normalization_53/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?decoder/decoder_block_18/batch_normalization_53/moving_variance
�
Sdecoder/decoder_block_18/batch_normalization_53/moving_variance/Read/ReadVariableOpReadVariableOp?decoder/decoder_block_18/batch_normalization_53/moving_variance*
_output_shapes
:@*
dtype0
�
;decoder/decoder_block_18/batch_normalization_53/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;decoder/decoder_block_18/batch_normalization_53/moving_mean
�
Odecoder/decoder_block_18/batch_normalization_53/moving_mean/Read/ReadVariableOpReadVariableOp;decoder/decoder_block_18/batch_normalization_53/moving_mean*
_output_shapes
:@*
dtype0
�
4decoder/decoder_block_18/batch_normalization_53/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64decoder/decoder_block_18/batch_normalization_53/beta
�
Hdecoder/decoder_block_18/batch_normalization_53/beta/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_18/batch_normalization_53/beta*
_output_shapes
:@*
dtype0
�
5decoder/decoder_block_18/batch_normalization_53/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75decoder/decoder_block_18/batch_normalization_53/gamma
�
Idecoder/decoder_block_18/batch_normalization_53/gamma/Read/ReadVariableOpReadVariableOp5decoder/decoder_block_18/batch_normalization_53/gamma*
_output_shapes
:@*
dtype0
�
'decoder/decoder_block_18/conv2d_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'decoder/decoder_block_18/conv2d_72/bias
�
;decoder/decoder_block_18/conv2d_72/bias/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_18/conv2d_72/bias*
_output_shapes
:@*
dtype0
�
)decoder/decoder_block_18/conv2d_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)decoder/decoder_block_18/conv2d_72/kernel
�
=decoder/decoder_block_18/conv2d_72/kernel/Read/ReadVariableOpReadVariableOp)decoder/decoder_block_18/conv2d_72/kernel*&
_output_shapes
:@@*
dtype0
�
decoder/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:� *&
shared_namedecoder/dense_20/bias
|
)decoder/dense_20/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_20/bias*
_output_shapes	
:� *
dtype0
�
decoder/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�� *(
shared_namedecoder/dense_20/kernel
�
+decoder/dense_20/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_20/kernel* 
_output_shapes
:
�� *
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1decoder/dense_20/kerneldecoder/dense_20/bias)decoder/decoder_block_18/conv2d_72/kernel'decoder/decoder_block_18/conv2d_72/bias5decoder/decoder_block_18/batch_normalization_53/gamma4decoder/decoder_block_18/batch_normalization_53/beta;decoder/decoder_block_18/batch_normalization_53/moving_mean?decoder/decoder_block_18/batch_normalization_53/moving_variance)decoder/decoder_block_19/conv2d_73/kernel'decoder/decoder_block_19/conv2d_73/bias5decoder/decoder_block_19/batch_normalization_54/gamma4decoder/decoder_block_19/batch_normalization_54/beta;decoder/decoder_block_19/batch_normalization_54/moving_mean?decoder/decoder_block_19/batch_normalization_54/moving_variance)decoder/decoder_block_20/conv2d_74/kernel'decoder/decoder_block_20/conv2d_74/bias5decoder/decoder_block_20/batch_normalization_55/gamma4decoder/decoder_block_20/batch_normalization_55/beta;decoder/decoder_block_20/batch_normalization_55/moving_mean?decoder/decoder_block_20/batch_normalization_55/moving_variancedecoder/conv2d_75/kerneldecoder/conv2d_75/biasdecoder/conv2d_76/kerneldecoder/conv2d_76/bias*$
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
%__inference_signature_wrapper_1917398

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
WQ
VARIABLE_VALUEdecoder/dense_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdecoder/dense_20/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)decoder/decoder_block_18/conv2d_72/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'decoder/decoder_block_18/conv2d_72/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5decoder/decoder_block_18/batch_normalization_53/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4decoder/decoder_block_18/batch_normalization_53/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;decoder/decoder_block_18/batch_normalization_53/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?decoder/decoder_block_18/batch_normalization_53/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)decoder/decoder_block_19/conv2d_73/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'decoder/decoder_block_19/conv2d_73/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5decoder/decoder_block_19/batch_normalization_54/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4decoder/decoder_block_19/batch_normalization_54/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;decoder/decoder_block_19/batch_normalization_54/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?decoder/decoder_block_19/batch_normalization_54/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)decoder/decoder_block_20/conv2d_74/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'decoder/decoder_block_20/conv2d_74/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5decoder/decoder_block_20/batch_normalization_55/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4decoder/decoder_block_20/batch_normalization_55/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;decoder/decoder_block_20/batch_normalization_55/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE?decoder/decoder_block_20/batch_normalization_55/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_75/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_75/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_76/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_76/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedecoder/dense_20/kerneldecoder/dense_20/bias)decoder/decoder_block_18/conv2d_72/kernel'decoder/decoder_block_18/conv2d_72/bias5decoder/decoder_block_18/batch_normalization_53/gamma4decoder/decoder_block_18/batch_normalization_53/beta;decoder/decoder_block_18/batch_normalization_53/moving_mean?decoder/decoder_block_18/batch_normalization_53/moving_variance)decoder/decoder_block_19/conv2d_73/kernel'decoder/decoder_block_19/conv2d_73/bias5decoder/decoder_block_19/batch_normalization_54/gamma4decoder/decoder_block_19/batch_normalization_54/beta;decoder/decoder_block_19/batch_normalization_54/moving_mean?decoder/decoder_block_19/batch_normalization_54/moving_variance)decoder/decoder_block_20/conv2d_74/kernel'decoder/decoder_block_20/conv2d_74/bias5decoder/decoder_block_20/batch_normalization_55/gamma4decoder/decoder_block_20/batch_normalization_55/beta;decoder/decoder_block_20/batch_normalization_55/moving_mean?decoder/decoder_block_20/batch_normalization_55/moving_variancedecoder/conv2d_75/kerneldecoder/conv2d_75/biasdecoder/conv2d_76/kerneldecoder/conv2d_76/biasConst*%
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
 __inference__traced_save_1918448
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder/dense_20/kerneldecoder/dense_20/bias)decoder/decoder_block_18/conv2d_72/kernel'decoder/decoder_block_18/conv2d_72/bias5decoder/decoder_block_18/batch_normalization_53/gamma4decoder/decoder_block_18/batch_normalization_53/beta;decoder/decoder_block_18/batch_normalization_53/moving_mean?decoder/decoder_block_18/batch_normalization_53/moving_variance)decoder/decoder_block_19/conv2d_73/kernel'decoder/decoder_block_19/conv2d_73/bias5decoder/decoder_block_19/batch_normalization_54/gamma4decoder/decoder_block_19/batch_normalization_54/beta;decoder/decoder_block_19/batch_normalization_54/moving_mean?decoder/decoder_block_19/batch_normalization_54/moving_variance)decoder/decoder_block_20/conv2d_74/kernel'decoder/decoder_block_20/conv2d_74/bias5decoder/decoder_block_20/batch_normalization_55/gamma4decoder/decoder_block_20/batch_normalization_55/beta;decoder/decoder_block_20/batch_normalization_55/moving_mean?decoder/decoder_block_20/batch_normalization_55/moving_variancedecoder/conv2d_75/kerneldecoder/conv2d_75/biasdecoder/conv2d_76/kerneldecoder/conv2d_76/bias*$
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
#__inference__traced_restore_1918530��
�
i
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1918219

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
�"
�
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916808
input_tensorB
(conv2d_73_conv2d_readvariableop_resource:@ 7
)conv2d_73_biasadd_readvariableop_resource: <
.batch_normalization_54_readvariableop_resource: >
0batch_normalization_54_readvariableop_1_resource: M
?batch_normalization_54_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: 
identity��6batch_normalization_54/FusedBatchNormV3/ReadVariableOp�8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_54/ReadVariableOp�'batch_normalization_54/ReadVariableOp_1� conv2d_73/BiasAdd/ReadVariableOp�conv2d_73/Conv2D/ReadVariableOp�
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_73/Conv2DConv2Dinput_tensor'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� l
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� g
up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_19/mulMulup_sampling2d_19/Const:output:0!up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_73/Relu:activations:0up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_54/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������   �
NoOpNoOp7^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_1!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�
�
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1916496

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
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
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918123

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
+__inference_conv2d_76_layer_call_fn_1918033

inputs!
unknown:
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1916721y
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
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
2__inference_decoder_block_20_layer_call_fn_1917946
input_tensor!
unknown: 
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916850y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������   : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������   
&
_user_specified_nameinput_tensor
�	
�
2__inference_decoder_block_18_layer_call_fn_1917745
input_tensor!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916593w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�
�
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1918024

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918105

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
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
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
̦
�
"__inference__wrapped_model_1916292
input_1C
/decoder_dense_20_matmul_readvariableop_resource:
�� ?
0decoder_dense_20_biasadd_readvariableop_resource:	� [
Adecoder_decoder_block_18_conv2d_72_conv2d_readvariableop_resource:@@P
Bdecoder_decoder_block_18_conv2d_72_biasadd_readvariableop_resource:@U
Gdecoder_decoder_block_18_batch_normalization_53_readvariableop_resource:@W
Idecoder_decoder_block_18_batch_normalization_53_readvariableop_1_resource:@f
Xdecoder_decoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@h
Zdecoder_decoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@[
Adecoder_decoder_block_19_conv2d_73_conv2d_readvariableop_resource:@ P
Bdecoder_decoder_block_19_conv2d_73_biasadd_readvariableop_resource: U
Gdecoder_decoder_block_19_batch_normalization_54_readvariableop_resource: W
Idecoder_decoder_block_19_batch_normalization_54_readvariableop_1_resource: f
Xdecoder_decoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resource: h
Zdecoder_decoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: [
Adecoder_decoder_block_20_conv2d_74_conv2d_readvariableop_resource: P
Bdecoder_decoder_block_20_conv2d_74_biasadd_readvariableop_resource:U
Gdecoder_decoder_block_20_batch_normalization_55_readvariableop_resource:W
Idecoder_decoder_block_20_batch_normalization_55_readvariableop_1_resource:f
Xdecoder_decoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resource:h
Zdecoder_decoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:J
0decoder_conv2d_75_conv2d_readvariableop_resource:?
1decoder_conv2d_75_biasadd_readvariableop_resource:J
0decoder_conv2d_76_conv2d_readvariableop_resource:?
1decoder_conv2d_76_biasadd_readvariableop_resource:
identity��(decoder/conv2d_75/BiasAdd/ReadVariableOp�'decoder/conv2d_75/Conv2D/ReadVariableOp�(decoder/conv2d_76/BiasAdd/ReadVariableOp�'decoder/conv2d_76/Conv2D/ReadVariableOp�Odecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp�Qdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�>decoder/decoder_block_18/batch_normalization_53/ReadVariableOp�@decoder/decoder_block_18/batch_normalization_53/ReadVariableOp_1�9decoder/decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp�8decoder/decoder_block_18/conv2d_72/Conv2D/ReadVariableOp�Odecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp�Qdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�>decoder/decoder_block_19/batch_normalization_54/ReadVariableOp�@decoder/decoder_block_19/batch_normalization_54/ReadVariableOp_1�9decoder/decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp�8decoder/decoder_block_19/conv2d_73/Conv2D/ReadVariableOp�Odecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp�Qdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�>decoder/decoder_block_20/batch_normalization_55/ReadVariableOp�@decoder/decoder_block_20/batch_normalization_55/ReadVariableOp_1�9decoder/decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp�8decoder/decoder_block_20/conv2d_74/Conv2D/ReadVariableOp�'decoder/dense_20/BiasAdd/ReadVariableOp�&decoder/dense_20/MatMul/ReadVariableOp�
&decoder/dense_20/MatMul/ReadVariableOpReadVariableOp/decoder_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0�
decoder/dense_20/MatMulMatMulinput_1.decoder/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
'decoder/dense_20/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
decoder/dense_20/BiasAddBiasAdd!decoder/dense_20/MatMul:product:0/decoder/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
decoder/dense_20/ReluRelu!decoder/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:���������� n
decoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
decoder/ReshapeReshape#decoder/dense_20/Relu:activations:0decoder/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
8decoder/decoder_block_18/conv2d_72/Conv2D/ReadVariableOpReadVariableOpAdecoder_decoder_block_18_conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
)decoder/decoder_block_18/conv2d_72/Conv2DConv2Ddecoder/Reshape:output:0@decoder/decoder_block_18/conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
9decoder/decoder_block_18/conv2d_72/BiasAdd/ReadVariableOpReadVariableOpBdecoder_decoder_block_18_conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
*decoder/decoder_block_18/conv2d_72/BiasAddBiasAdd2decoder/decoder_block_18/conv2d_72/Conv2D:output:0Adecoder/decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
'decoder/decoder_block_18/conv2d_72/ReluRelu3decoder/decoder_block_18/conv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
/decoder/decoder_block_18/up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
1decoder/decoder_block_18/up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
-decoder/decoder_block_18/up_sampling2d_18/mulMul8decoder/decoder_block_18/up_sampling2d_18/Const:output:0:decoder/decoder_block_18/up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
Fdecoder/decoder_block_18/up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighbor5decoder/decoder_block_18/conv2d_72/Relu:activations:01decoder/decoder_block_18/up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
>decoder/decoder_block_18/batch_normalization_53/ReadVariableOpReadVariableOpGdecoder_decoder_block_18_batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
@decoder/decoder_block_18/batch_normalization_53/ReadVariableOp_1ReadVariableOpIdecoder_decoder_block_18_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Odecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpXdecoder_decoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Qdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZdecoder_decoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
@decoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3FusedBatchNormV3Wdecoder/decoder_block_18/up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0Fdecoder/decoder_block_18/batch_normalization_53/ReadVariableOp:value:0Hdecoder/decoder_block_18/batch_normalization_53/ReadVariableOp_1:value:0Wdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0Ydecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
8decoder/decoder_block_19/conv2d_73/Conv2D/ReadVariableOpReadVariableOpAdecoder_decoder_block_19_conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
)decoder/decoder_block_19/conv2d_73/Conv2DConv2DDdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3:y:0@decoder/decoder_block_19/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
9decoder/decoder_block_19/conv2d_73/BiasAdd/ReadVariableOpReadVariableOpBdecoder_decoder_block_19_conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
*decoder/decoder_block_19/conv2d_73/BiasAddBiasAdd2decoder/decoder_block_19/conv2d_73/Conv2D:output:0Adecoder/decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
'decoder/decoder_block_19/conv2d_73/ReluRelu3decoder/decoder_block_19/conv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� �
/decoder/decoder_block_19/up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      �
1decoder/decoder_block_19/up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
-decoder/decoder_block_19/up_sampling2d_19/mulMul8decoder/decoder_block_19/up_sampling2d_19/Const:output:0:decoder/decoder_block_19/up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
Fdecoder/decoder_block_19/up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighbor5decoder/decoder_block_19/conv2d_73/Relu:activations:01decoder/decoder_block_19/up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
>decoder/decoder_block_19/batch_normalization_54/ReadVariableOpReadVariableOpGdecoder_decoder_block_19_batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
@decoder/decoder_block_19/batch_normalization_54/ReadVariableOp_1ReadVariableOpIdecoder_decoder_block_19_batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Odecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOpXdecoder_decoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Qdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZdecoder_decoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
@decoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3FusedBatchNormV3Wdecoder/decoder_block_19/up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0Fdecoder/decoder_block_19/batch_normalization_54/ReadVariableOp:value:0Hdecoder/decoder_block_19/batch_normalization_54/ReadVariableOp_1:value:0Wdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0Ydecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
8decoder/decoder_block_20/conv2d_74/Conv2D/ReadVariableOpReadVariableOpAdecoder_decoder_block_20_conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
)decoder/decoder_block_20/conv2d_74/Conv2DConv2DDdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3:y:0@decoder/decoder_block_20/conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
9decoder/decoder_block_20/conv2d_74/BiasAdd/ReadVariableOpReadVariableOpBdecoder_decoder_block_20_conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
*decoder/decoder_block_20/conv2d_74/BiasAddBiasAdd2decoder/decoder_block_20/conv2d_74/Conv2D:output:0Adecoder/decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
'decoder/decoder_block_20/conv2d_74/ReluRelu3decoder/decoder_block_20/conv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  �
/decoder/decoder_block_20/up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        �
1decoder/decoder_block_20/up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
-decoder/decoder_block_20/up_sampling2d_20/mulMul8decoder/decoder_block_20/up_sampling2d_20/Const:output:0:decoder/decoder_block_20/up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
Fdecoder/decoder_block_20/up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighbor5decoder/decoder_block_20/conv2d_74/Relu:activations:01decoder/decoder_block_20/up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
>decoder/decoder_block_20/batch_normalization_55/ReadVariableOpReadVariableOpGdecoder_decoder_block_20_batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
@decoder/decoder_block_20/batch_normalization_55/ReadVariableOp_1ReadVariableOpIdecoder_decoder_block_20_batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Odecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpXdecoder_decoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Qdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZdecoder_decoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
@decoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3Wdecoder/decoder_block_20/up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0Fdecoder/decoder_block_20/batch_normalization_55/ReadVariableOp:value:0Hdecoder/decoder_block_20/batch_normalization_55/ReadVariableOp_1:value:0Wdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0Ydecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
'decoder/conv2d_75/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
decoder/conv2d_75/Conv2DConv2DDdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3:y:0/decoder/conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
(decoder/conv2d_75/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
decoder/conv2d_75/BiasAddBiasAdd!decoder/conv2d_75/Conv2D:output:00decoder/conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������~
decoder/conv2d_75/ReluRelu"decoder/conv2d_75/BiasAdd:output:0*
T0*1
_output_shapes
:������������
'decoder/conv2d_76/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_76_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
decoder/conv2d_76/Conv2DConv2D$decoder/conv2d_75/Relu:activations:0/decoder/conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
(decoder/conv2d_76/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
decoder/conv2d_76/BiasAddBiasAdd!decoder/conv2d_76/Conv2D:output:00decoder/conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
decoder/conv2d_76/SigmoidSigmoid"decoder/conv2d_76/BiasAdd:output:0*
T0*1
_output_shapes
:�����������v
IdentityIdentitydecoder/conv2d_76/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp)^decoder/conv2d_75/BiasAdd/ReadVariableOp(^decoder/conv2d_75/Conv2D/ReadVariableOp)^decoder/conv2d_76/BiasAdd/ReadVariableOp(^decoder/conv2d_76/Conv2D/ReadVariableOpP^decoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpR^decoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?^decoder/decoder_block_18/batch_normalization_53/ReadVariableOpA^decoder/decoder_block_18/batch_normalization_53/ReadVariableOp_1:^decoder/decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp9^decoder/decoder_block_18/conv2d_72/Conv2D/ReadVariableOpP^decoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpR^decoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1?^decoder/decoder_block_19/batch_normalization_54/ReadVariableOpA^decoder/decoder_block_19/batch_normalization_54/ReadVariableOp_1:^decoder/decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp9^decoder/decoder_block_19/conv2d_73/Conv2D/ReadVariableOpP^decoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpR^decoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1?^decoder/decoder_block_20/batch_normalization_55/ReadVariableOpA^decoder/decoder_block_20/batch_normalization_55/ReadVariableOp_1:^decoder/decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp9^decoder/decoder_block_20/conv2d_74/Conv2D/ReadVariableOp(^decoder/dense_20/BiasAdd/ReadVariableOp'^decoder/dense_20/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2T
(decoder/conv2d_75/BiasAdd/ReadVariableOp(decoder/conv2d_75/BiasAdd/ReadVariableOp2R
'decoder/conv2d_75/Conv2D/ReadVariableOp'decoder/conv2d_75/Conv2D/ReadVariableOp2T
(decoder/conv2d_76/BiasAdd/ReadVariableOp(decoder/conv2d_76/BiasAdd/ReadVariableOp2R
'decoder/conv2d_76/Conv2D/ReadVariableOp'decoder/conv2d_76/Conv2D/ReadVariableOp2�
Qdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1Qdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12�
Odecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpOdecoder/decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2�
@decoder/decoder_block_18/batch_normalization_53/ReadVariableOp_1@decoder/decoder_block_18/batch_normalization_53/ReadVariableOp_12�
>decoder/decoder_block_18/batch_normalization_53/ReadVariableOp>decoder/decoder_block_18/batch_normalization_53/ReadVariableOp2v
9decoder/decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp9decoder/decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp2t
8decoder/decoder_block_18/conv2d_72/Conv2D/ReadVariableOp8decoder/decoder_block_18/conv2d_72/Conv2D/ReadVariableOp2�
Qdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1Qdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12�
Odecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpOdecoder/decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp2�
@decoder/decoder_block_19/batch_normalization_54/ReadVariableOp_1@decoder/decoder_block_19/batch_normalization_54/ReadVariableOp_12�
>decoder/decoder_block_19/batch_normalization_54/ReadVariableOp>decoder/decoder_block_19/batch_normalization_54/ReadVariableOp2v
9decoder/decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp9decoder/decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp2t
8decoder/decoder_block_19/conv2d_73/Conv2D/ReadVariableOp8decoder/decoder_block_19/conv2d_73/Conv2D/ReadVariableOp2�
Qdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Qdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12�
Odecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpOdecoder/decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2�
@decoder/decoder_block_20/batch_normalization_55/ReadVariableOp_1@decoder/decoder_block_20/batch_normalization_55/ReadVariableOp_12�
>decoder/decoder_block_20/batch_normalization_55/ReadVariableOp>decoder/decoder_block_20/batch_normalization_55/ReadVariableOp2v
9decoder/decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp9decoder/decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp2t
8decoder/decoder_block_20/conv2d_74/Conv2D/ReadVariableOp8decoder/decoder_block_20/conv2d_74/Conv2D/ReadVariableOp2R
'decoder/dense_20/BiasAdd/ReadVariableOp'decoder/dense_20/BiasAdd/ReadVariableOp2P
&decoder/dense_20/MatMul/ReadVariableOp&decoder/dense_20/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1916721

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1916388

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
8__inference_batch_normalization_54_layer_call_fn_1918153

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1916413�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
+__inference_conv2d_75_layer_call_fn_1918013

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1916704y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_1917728

inputs2
matmul_readvariableop_resource:
�� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:���������� b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:���������� w
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
�
�
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918184

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
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
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_54_layer_call_fn_1918166

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1916431�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1917451
embedding_input
unknown:
�� 
	unknown_0:	� #
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@#
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

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
D__inference_decoder_layer_call_and_return_conditional_losses_1916938y
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
�
N
2__inference_up_sampling2d_18_layer_call_fn_1918049

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
GPU2 *0J 8� *V
fQRO
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1916305�
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
i
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1916305

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
�
)__inference_decoder_layer_call_fn_1917504
embedding_input
unknown:
�� 
	unknown_0:	� #
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@#
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

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
D__inference_decoder_layer_call_and_return_conditional_losses_1917051y
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
�
�
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918263

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
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
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�*
�
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916679
input_tensorB
(conv2d_74_conv2d_readvariableop_resource: 7
)conv2d_74_biasadd_readvariableop_resource:<
.batch_normalization_55_readvariableop_resource:>
0batch_normalization_55_readvariableop_1_resource:M
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:
identity��%batch_normalization_55/AssignNewValue�'batch_normalization_55/AssignNewValue_1�6batch_normalization_55/FusedBatchNormV3/ReadVariableOp�8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_55/ReadVariableOp�'batch_normalization_55/ReadVariableOp_1� conv2d_74/BiasAdd/ReadVariableOp�conv2d_74/Conv2D/ReadVariableOp�
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_74/Conv2DConv2Dinput_tensor'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  l
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  g
up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        i
up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_20/mulMulup_sampling2d_20/Const:output:0!up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_74/Relu:activations:0up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_55/AssignNewValueAssignVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource4batch_normalization_55/FusedBatchNormV3:batch_mean:07^batch_normalization_55/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_55/AssignNewValue_1AssignVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_55/FusedBatchNormV3:batch_variance:09^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_55/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp&^batch_normalization_55/AssignNewValue(^batch_normalization_55/AssignNewValue_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������   : : : : : : 2R
'batch_normalization_55/AssignNewValue_1'batch_normalization_55/AssignNewValue_12N
%batch_normalization_55/AssignNewValue%batch_normalization_55/AssignNewValue2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������   
&
_user_specified_nameinput_tensor
ɖ
�
D__inference_decoder_layer_call_and_return_conditional_losses_1917708
embedding_input;
'dense_20_matmul_readvariableop_resource:
�� 7
(dense_20_biasadd_readvariableop_resource:	� S
9decoder_block_18_conv2d_72_conv2d_readvariableop_resource:@@H
:decoder_block_18_conv2d_72_biasadd_readvariableop_resource:@M
?decoder_block_18_batch_normalization_53_readvariableop_resource:@O
Adecoder_block_18_batch_normalization_53_readvariableop_1_resource:@^
Pdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@`
Rdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@S
9decoder_block_19_conv2d_73_conv2d_readvariableop_resource:@ H
:decoder_block_19_conv2d_73_biasadd_readvariableop_resource: M
?decoder_block_19_batch_normalization_54_readvariableop_resource: O
Adecoder_block_19_batch_normalization_54_readvariableop_1_resource: ^
Pdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resource: `
Rdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: S
9decoder_block_20_conv2d_74_conv2d_readvariableop_resource: H
:decoder_block_20_conv2d_74_biasadd_readvariableop_resource:M
?decoder_block_20_batch_normalization_55_readvariableop_resource:O
Adecoder_block_20_batch_normalization_55_readvariableop_1_resource:^
Pdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resource:`
Rdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_75_conv2d_readvariableop_resource:7
)conv2d_75_biasadd_readvariableop_resource:B
(conv2d_76_conv2d_readvariableop_resource:7
)conv2d_76_biasadd_readvariableop_resource:
identity�� conv2d_75/BiasAdd/ReadVariableOp�conv2d_75/Conv2D/ReadVariableOp� conv2d_76/BiasAdd/ReadVariableOp�conv2d_76/Conv2D/ReadVariableOp�Gdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp�Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�6decoder_block_18/batch_normalization_53/ReadVariableOp�8decoder_block_18/batch_normalization_53/ReadVariableOp_1�1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp�0decoder_block_18/conv2d_72/Conv2D/ReadVariableOp�Gdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp�Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�6decoder_block_19/batch_normalization_54/ReadVariableOp�8decoder_block_19/batch_normalization_54/ReadVariableOp_1�1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp�0decoder_block_19/conv2d_73/Conv2D/ReadVariableOp�Gdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp�Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�6decoder_block_20/batch_normalization_55/ReadVariableOp�8decoder_block_20/batch_normalization_55/ReadVariableOp_1�1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp�0decoder_block_20/conv2d_74/Conv2D/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0�
dense_20/MatMulMatMulembedding_input&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:���������� f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
ReshapeReshapedense_20/Relu:activations:0Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
0decoder_block_18/conv2d_72/Conv2D/ReadVariableOpReadVariableOp9decoder_block_18_conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
!decoder_block_18/conv2d_72/Conv2DConv2DReshape:output:08decoder_block_18/conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_18_conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
"decoder_block_18/conv2d_72/BiasAddBiasAdd*decoder_block_18/conv2d_72/Conv2D:output:09decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
decoder_block_18/conv2d_72/ReluRelu+decoder_block_18/conv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@x
'decoder_block_18/up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      z
)decoder_block_18/up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
%decoder_block_18/up_sampling2d_18/mulMul0decoder_block_18/up_sampling2d_18/Const:output:02decoder_block_18/up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
>decoder_block_18/up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighbor-decoder_block_18/conv2d_72/Relu:activations:0)decoder_block_18/up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
6decoder_block_18/batch_normalization_53/ReadVariableOpReadVariableOp?decoder_block_18_batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
8decoder_block_18/batch_normalization_53/ReadVariableOp_1ReadVariableOpAdecoder_block_18_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Gdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
8decoder_block_18/batch_normalization_53/FusedBatchNormV3FusedBatchNormV3Odecoder_block_18/up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0>decoder_block_18/batch_normalization_53/ReadVariableOp:value:0@decoder_block_18/batch_normalization_53/ReadVariableOp_1:value:0Odecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
0decoder_block_19/conv2d_73/Conv2D/ReadVariableOpReadVariableOp9decoder_block_19_conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
!decoder_block_19/conv2d_73/Conv2DConv2D<decoder_block_18/batch_normalization_53/FusedBatchNormV3:y:08decoder_block_19/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_19_conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"decoder_block_19/conv2d_73/BiasAddBiasAdd*decoder_block_19/conv2d_73/Conv2D:output:09decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
decoder_block_19/conv2d_73/ReluRelu+decoder_block_19/conv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� x
'decoder_block_19/up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      z
)decoder_block_19/up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
%decoder_block_19/up_sampling2d_19/mulMul0decoder_block_19/up_sampling2d_19/Const:output:02decoder_block_19/up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
>decoder_block_19/up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighbor-decoder_block_19/conv2d_73/Relu:activations:0)decoder_block_19/up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
6decoder_block_19/batch_normalization_54/ReadVariableOpReadVariableOp?decoder_block_19_batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
8decoder_block_19/batch_normalization_54/ReadVariableOp_1ReadVariableOpAdecoder_block_19_batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Gdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8decoder_block_19/batch_normalization_54/FusedBatchNormV3FusedBatchNormV3Odecoder_block_19/up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0>decoder_block_19/batch_normalization_54/ReadVariableOp:value:0@decoder_block_19/batch_normalization_54/ReadVariableOp_1:value:0Odecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
0decoder_block_20/conv2d_74/Conv2D/ReadVariableOpReadVariableOp9decoder_block_20_conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!decoder_block_20/conv2d_74/Conv2DConv2D<decoder_block_19/batch_normalization_54/FusedBatchNormV3:y:08decoder_block_20/conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_20_conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"decoder_block_20/conv2d_74/BiasAddBiasAdd*decoder_block_20/conv2d_74/Conv2D:output:09decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
decoder_block_20/conv2d_74/ReluRelu+decoder_block_20/conv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  x
'decoder_block_20/up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        z
)decoder_block_20/up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
%decoder_block_20/up_sampling2d_20/mulMul0decoder_block_20/up_sampling2d_20/Const:output:02decoder_block_20/up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
>decoder_block_20/up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighbor-decoder_block_20/conv2d_74/Relu:activations:0)decoder_block_20/up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
6decoder_block_20/batch_normalization_55/ReadVariableOpReadVariableOp?decoder_block_20_batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
8decoder_block_20/batch_normalization_55/ReadVariableOp_1ReadVariableOpAdecoder_block_20_batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Gdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
8decoder_block_20/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3Odecoder_block_20/up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0>decoder_block_20/batch_normalization_55/ReadVariableOp:value:0@decoder_block_20/batch_normalization_55/ReadVariableOp_1:value:0Odecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
conv2d_75/Conv2D/ReadVariableOpReadVariableOp(conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_75/Conv2DConv2D<decoder_block_20/batch_normalization_55/FusedBatchNormV3:y:0'conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_75/BiasAdd/ReadVariableOpReadVariableOp)conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_75/BiasAddBiasAddconv2d_75/Conv2D:output:0(conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������n
conv2d_75/ReluReluconv2d_75/BiasAdd:output:0*
T0*1
_output_shapes
:������������
conv2d_76/Conv2D/ReadVariableOpReadVariableOp(conv2d_76_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_76/Conv2DConv2Dconv2d_75/Relu:activations:0'conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_76/BiasAdd/ReadVariableOpReadVariableOp)conv2d_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_76/BiasAddBiasAddconv2d_76/Conv2D:output:0(conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_76/SigmoidSigmoidconv2d_76/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
IdentityIdentityconv2d_76/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������

NoOpNoOp!^conv2d_75/BiasAdd/ReadVariableOp ^conv2d_75/Conv2D/ReadVariableOp!^conv2d_76/BiasAdd/ReadVariableOp ^conv2d_76/Conv2D/ReadVariableOpH^decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpJ^decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_17^decoder_block_18/batch_normalization_53/ReadVariableOp9^decoder_block_18/batch_normalization_53/ReadVariableOp_12^decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp1^decoder_block_18/conv2d_72/Conv2D/ReadVariableOpH^decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpJ^decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_17^decoder_block_19/batch_normalization_54/ReadVariableOp9^decoder_block_19/batch_normalization_54/ReadVariableOp_12^decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp1^decoder_block_19/conv2d_73/Conv2D/ReadVariableOpH^decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpJ^decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_17^decoder_block_20/batch_normalization_55/ReadVariableOp9^decoder_block_20/batch_normalization_55/ReadVariableOp_12^decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp1^decoder_block_20/conv2d_74/Conv2D/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_75/BiasAdd/ReadVariableOp conv2d_75/BiasAdd/ReadVariableOp2B
conv2d_75/Conv2D/ReadVariableOpconv2d_75/Conv2D/ReadVariableOp2D
 conv2d_76/BiasAdd/ReadVariableOp conv2d_76/BiasAdd/ReadVariableOp2B
conv2d_76/Conv2D/ReadVariableOpconv2d_76/Conv2D/ReadVariableOp2�
Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12�
Gdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpGdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_18/batch_normalization_53/ReadVariableOp_18decoder_block_18/batch_normalization_53/ReadVariableOp_12p
6decoder_block_18/batch_normalization_53/ReadVariableOp6decoder_block_18/batch_normalization_53/ReadVariableOp2f
1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp2d
0decoder_block_18/conv2d_72/Conv2D/ReadVariableOp0decoder_block_18/conv2d_72/Conv2D/ReadVariableOp2�
Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12�
Gdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpGdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_19/batch_normalization_54/ReadVariableOp_18decoder_block_19/batch_normalization_54/ReadVariableOp_12p
6decoder_block_19/batch_normalization_54/ReadVariableOp6decoder_block_19/batch_normalization_54/ReadVariableOp2f
1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp2d
0decoder_block_19/conv2d_73/Conv2D/ReadVariableOp0decoder_block_19/conv2d_73/Conv2D/ReadVariableOp2�
Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12�
Gdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpGdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_20/batch_normalization_55/ReadVariableOp_18decoder_block_20/batch_normalization_55/ReadVariableOp_12p
6decoder_block_20/batch_normalization_55/ReadVariableOp6decoder_block_20/batch_normalization_55/ReadVariableOp2f
1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp2d
0decoder_block_20/conv2d_74/Conv2D/ReadVariableOp0decoder_block_20/conv2d_74/Conv2D/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�*
�
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917883
input_tensorB
(conv2d_73_conv2d_readvariableop_resource:@ 7
)conv2d_73_biasadd_readvariableop_resource: <
.batch_normalization_54_readvariableop_resource: >
0batch_normalization_54_readvariableop_1_resource: M
?batch_normalization_54_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: 
identity��%batch_normalization_54/AssignNewValue�'batch_normalization_54/AssignNewValue_1�6batch_normalization_54/FusedBatchNormV3/ReadVariableOp�8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_54/ReadVariableOp�'batch_normalization_54/ReadVariableOp_1� conv2d_73/BiasAdd/ReadVariableOp�conv2d_73/Conv2D/ReadVariableOp�
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_73/Conv2DConv2Dinput_tensor'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� l
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� g
up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_19/mulMulup_sampling2d_19/Const:output:0!up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_73/Relu:activations:0up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_54/AssignNewValueAssignVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource4batch_normalization_54/FusedBatchNormV3:batch_mean:07^batch_normalization_54/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_54/AssignNewValue_1AssignVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_54/FusedBatchNormV3:batch_variance:09^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_54/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������   �
NoOpNoOp&^batch_normalization_54/AssignNewValue(^batch_normalization_54/AssignNewValue_17^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_1!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2R
'batch_normalization_54/AssignNewValue_1'batch_normalization_54/AssignNewValue_12N
%batch_normalization_54/AssignNewValue%batch_normalization_54/AssignNewValue2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�
�
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1918044

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1916431

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�,
�

D__inference_decoder_layer_call_and_return_conditional_losses_1917051
embedding_input$
dense_20_1916994:
�� 
dense_20_1916996:	� 2
decoder_block_18_1917001:@@&
decoder_block_18_1917003:@&
decoder_block_18_1917005:@&
decoder_block_18_1917007:@&
decoder_block_18_1917009:@&
decoder_block_18_1917011:@2
decoder_block_19_1917014:@ &
decoder_block_19_1917016: &
decoder_block_19_1917018: &
decoder_block_19_1917020: &
decoder_block_19_1917022: &
decoder_block_19_1917024: 2
decoder_block_20_1917027: &
decoder_block_20_1917029:&
decoder_block_20_1917031:&
decoder_block_20_1917033:&
decoder_block_20_1917035:&
decoder_block_20_1917037:+
conv2d_75_1917040:
conv2d_75_1917042:+
conv2d_76_1917045:
conv2d_76_1917047:
identity��!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�(decoder_block_18/StatefulPartitionedCall�(decoder_block_19/StatefulPartitionedCall�(decoder_block_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_20_1916994dense_20_1916996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1916556f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
ReshapeReshape)dense_20/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_18_1917001decoder_block_18_1917003decoder_block_18_1917005decoder_block_18_1917007decoder_block_18_1917009decoder_block_18_1917011*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916766�
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0decoder_block_19_1917014decoder_block_19_1917016decoder_block_19_1917018decoder_block_19_1917020decoder_block_19_1917022decoder_block_19_1917024*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916808�
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0decoder_block_20_1917027decoder_block_20_1917029decoder_block_20_1917031decoder_block_20_1917033decoder_block_20_1917035decoder_block_20_1917037*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916850�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_75_1917040conv2d_75_1917042*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1916704�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0conv2d_76_1917045conv2d_76_1917047*
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1916721�
IdentityIdentity*conv2d_76/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
i
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1916471

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
�
i
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1918140

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
8__inference_batch_normalization_55_layer_call_fn_1918245

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1916514�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_53_layer_call_fn_1918074

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1916330�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
N
2__inference_up_sampling2d_20_layer_call_fn_1918207

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
GPU2 *0J 8� *V
fQRO
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1916471�
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
�,
�

D__inference_decoder_layer_call_and_return_conditional_losses_1916938
embedding_input$
dense_20_1916881:
�� 
dense_20_1916883:	� 2
decoder_block_18_1916888:@@&
decoder_block_18_1916890:@&
decoder_block_18_1916892:@&
decoder_block_18_1916894:@&
decoder_block_18_1916896:@&
decoder_block_18_1916898:@2
decoder_block_19_1916901:@ &
decoder_block_19_1916903: &
decoder_block_19_1916905: &
decoder_block_19_1916907: &
decoder_block_19_1916909: &
decoder_block_19_1916911: 2
decoder_block_20_1916914: &
decoder_block_20_1916916:&
decoder_block_20_1916918:&
decoder_block_20_1916920:&
decoder_block_20_1916922:&
decoder_block_20_1916924:+
conv2d_75_1916927:
conv2d_75_1916929:+
conv2d_76_1916932:
conv2d_76_1916934:
identity��!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�(decoder_block_18/StatefulPartitionedCall�(decoder_block_19/StatefulPartitionedCall�(decoder_block_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_20_1916881dense_20_1916883*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1916556f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
ReshapeReshape)dense_20/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_18_1916888decoder_block_18_1916890decoder_block_18_1916892decoder_block_18_1916894decoder_block_18_1916896decoder_block_18_1916898*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916593�
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0decoder_block_19_1916901decoder_block_19_1916903decoder_block_19_1916905decoder_block_19_1916907decoder_block_19_1916909decoder_block_19_1916911*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916636�
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0decoder_block_20_1916914decoder_block_20_1916916decoder_block_20_1916918decoder_block_20_1916920decoder_block_20_1916922decoder_block_20_1916924*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916679�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_75_1916927conv2d_75_1916929*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1916704�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0conv2d_76_1916932conv2d_76_1916934*
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1916721�
IdentityIdentity*conv2d_76/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
�
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1916330

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
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
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1916989
input_1
unknown:
�� 
	unknown_0:	� #
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@#
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

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
D__inference_decoder_layer_call_and_return_conditional_losses_1916938y
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
�"
�
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917820
input_tensorB
(conv2d_72_conv2d_readvariableop_resource:@@7
)conv2d_72_biasadd_readvariableop_resource:@<
.batch_normalization_53_readvariableop_resource:@>
0batch_normalization_53_readvariableop_1_resource:@M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@
identity��6batch_normalization_53/FusedBatchNormV3/ReadVariableOp�8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_53/ReadVariableOp�'batch_normalization_53/ReadVariableOp_1� conv2d_72/BiasAdd/ReadVariableOp�conv2d_72/Conv2D/ReadVariableOp�
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_72/Conv2DConv2Dinput_tensor'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_18/mulMulup_sampling2d_18/Const:output:0!up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_72/Relu:activations:0up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_53/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp7^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�	
�
2__inference_decoder_block_20_layer_call_fn_1917929
input_tensor!
unknown: 
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916679y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������   : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������   
&
_user_specified_nameinput_tensor
�
i
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1918061

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
�"
�
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916850
input_tensorB
(conv2d_74_conv2d_readvariableop_resource: 7
)conv2d_74_biasadd_readvariableop_resource:<
.batch_normalization_55_readvariableop_resource:>
0batch_normalization_55_readvariableop_1_resource:M
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:
identity��6batch_normalization_55/FusedBatchNormV3/ReadVariableOp�8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_55/ReadVariableOp�'batch_normalization_55/ReadVariableOp_1� conv2d_74/BiasAdd/ReadVariableOp�conv2d_74/Conv2D/ReadVariableOp�
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_74/Conv2DConv2Dinput_tensor'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  l
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  g
up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        i
up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_20/mulMulup_sampling2d_20/Const:output:0!up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_74/Relu:activations:0up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_55/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp7^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������   : : : : : : 2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������   
&
_user_specified_nameinput_tensor
��
�
 __inference__traced_save_1918448
file_prefixB
.read_disablecopyonread_decoder_dense_20_kernel:
�� =
.read_1_disablecopyonread_decoder_dense_20_bias:	� \
Bread_2_disablecopyonread_decoder_decoder_block_18_conv2d_72_kernel:@@N
@read_3_disablecopyonread_decoder_decoder_block_18_conv2d_72_bias:@\
Nread_4_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_gamma:@[
Mread_5_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_beta:@b
Tread_6_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_moving_mean:@f
Xread_7_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_moving_variance:@\
Bread_8_disablecopyonread_decoder_decoder_block_19_conv2d_73_kernel:@ N
@read_9_disablecopyonread_decoder_decoder_block_19_conv2d_73_bias: ]
Oread_10_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_gamma: \
Nread_11_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_beta: c
Uread_12_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_moving_mean: g
Yread_13_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_moving_variance: ]
Cread_14_disablecopyonread_decoder_decoder_block_20_conv2d_74_kernel: O
Aread_15_disablecopyonread_decoder_decoder_block_20_conv2d_74_bias:]
Oread_16_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_gamma:\
Nread_17_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_beta:c
Uread_18_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_moving_mean:g
Yread_19_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_moving_variance:L
2read_20_disablecopyonread_decoder_conv2d_75_kernel:>
0read_21_disablecopyonread_decoder_conv2d_75_bias:L
2read_22_disablecopyonread_decoder_conv2d_76_kernel:>
0read_23_disablecopyonread_decoder_conv2d_76_bias:
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
: �
Read/DisableCopyOnReadDisableCopyOnRead.read_disablecopyonread_decoder_dense_20_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp.read_disablecopyonread_decoder_dense_20_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
�� *
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
�� c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�� �
Read_1/DisableCopyOnReadDisableCopyOnRead.read_1_disablecopyonread_decoder_dense_20_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp.read_1_disablecopyonread_decoder_dense_20_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:� *
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:� `

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:� �
Read_2/DisableCopyOnReadDisableCopyOnReadBread_2_disablecopyonread_decoder_decoder_block_18_conv2d_72_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOpBread_2_disablecopyonread_decoder_decoder_block_18_conv2d_72_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_3/DisableCopyOnReadDisableCopyOnRead@read_3_disablecopyonread_decoder_decoder_block_18_conv2d_72_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp@read_3_disablecopyonread_decoder_decoder_block_18_conv2d_72_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_4/DisableCopyOnReadDisableCopyOnReadNread_4_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_gamma"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpNread_4_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_5/DisableCopyOnReadDisableCopyOnReadMread_5_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_beta"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpMread_5_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_6/DisableCopyOnReadDisableCopyOnReadTread_6_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpTread_6_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_7/DisableCopyOnReadDisableCopyOnReadXread_7_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpXread_7_disablecopyonread_decoder_decoder_block_18_batch_normalization_53_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_8/DisableCopyOnReadDisableCopyOnReadBread_8_disablecopyonread_decoder_decoder_block_19_conv2d_73_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOpBread_8_disablecopyonread_decoder_decoder_block_19_conv2d_73_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0v
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_9/DisableCopyOnReadDisableCopyOnRead@read_9_disablecopyonread_decoder_decoder_block_19_conv2d_73_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp@read_9_disablecopyonread_decoder_decoder_block_19_conv2d_73_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnReadOread_10_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpOread_10_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnReadNread_11_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpNread_11_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnReadUread_12_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpUread_12_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_13/DisableCopyOnReadDisableCopyOnReadYread_13_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpYread_13_disablecopyonread_decoder_decoder_block_19_batch_normalization_54_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnReadCread_14_disablecopyonread_decoder_decoder_block_20_conv2d_74_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpCread_14_disablecopyonread_decoder_decoder_block_20_conv2d_74_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnReadAread_15_disablecopyonread_decoder_decoder_block_20_conv2d_74_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpAread_15_disablecopyonread_decoder_decoder_block_20_conv2d_74_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnReadOread_16_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpOread_16_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnReadNread_17_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpNread_17_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_18/DisableCopyOnReadDisableCopyOnReadUread_18_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpUread_18_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnReadYread_19_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpYread_19_disablecopyonread_decoder_decoder_block_20_batch_normalization_55_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_decoder_conv2d_75_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_decoder_conv2d_75_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_decoder_conv2d_75_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_decoder_conv2d_75_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_decoder_conv2d_76_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_decoder_conv2d_76_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_decoder_conv2d_76_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_decoder_conv2d_76_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
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
��
�
D__inference_decoder_layer_call_and_return_conditional_losses_1917606
embedding_input;
'dense_20_matmul_readvariableop_resource:
�� 7
(dense_20_biasadd_readvariableop_resource:	� S
9decoder_block_18_conv2d_72_conv2d_readvariableop_resource:@@H
:decoder_block_18_conv2d_72_biasadd_readvariableop_resource:@M
?decoder_block_18_batch_normalization_53_readvariableop_resource:@O
Adecoder_block_18_batch_normalization_53_readvariableop_1_resource:@^
Pdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@`
Rdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@S
9decoder_block_19_conv2d_73_conv2d_readvariableop_resource:@ H
:decoder_block_19_conv2d_73_biasadd_readvariableop_resource: M
?decoder_block_19_batch_normalization_54_readvariableop_resource: O
Adecoder_block_19_batch_normalization_54_readvariableop_1_resource: ^
Pdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resource: `
Rdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: S
9decoder_block_20_conv2d_74_conv2d_readvariableop_resource: H
:decoder_block_20_conv2d_74_biasadd_readvariableop_resource:M
?decoder_block_20_batch_normalization_55_readvariableop_resource:O
Adecoder_block_20_batch_normalization_55_readvariableop_1_resource:^
Pdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resource:`
Rdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_75_conv2d_readvariableop_resource:7
)conv2d_75_biasadd_readvariableop_resource:B
(conv2d_76_conv2d_readvariableop_resource:7
)conv2d_76_biasadd_readvariableop_resource:
identity�� conv2d_75/BiasAdd/ReadVariableOp�conv2d_75/Conv2D/ReadVariableOp� conv2d_76/BiasAdd/ReadVariableOp�conv2d_76/Conv2D/ReadVariableOp�6decoder_block_18/batch_normalization_53/AssignNewValue�8decoder_block_18/batch_normalization_53/AssignNewValue_1�Gdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp�Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�6decoder_block_18/batch_normalization_53/ReadVariableOp�8decoder_block_18/batch_normalization_53/ReadVariableOp_1�1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp�0decoder_block_18/conv2d_72/Conv2D/ReadVariableOp�6decoder_block_19/batch_normalization_54/AssignNewValue�8decoder_block_19/batch_normalization_54/AssignNewValue_1�Gdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp�Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�6decoder_block_19/batch_normalization_54/ReadVariableOp�8decoder_block_19/batch_normalization_54/ReadVariableOp_1�1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp�0decoder_block_19/conv2d_73/Conv2D/ReadVariableOp�6decoder_block_20/batch_normalization_55/AssignNewValue�8decoder_block_20/batch_normalization_55/AssignNewValue_1�Gdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp�Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�6decoder_block_20/batch_normalization_55/ReadVariableOp�8decoder_block_20/batch_normalization_55/ReadVariableOp_1�1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp�0decoder_block_20/conv2d_74/Conv2D/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0�
dense_20/MatMulMatMulembedding_input&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� �
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:���������� f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
ReshapeReshapedense_20/Relu:activations:0Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
0decoder_block_18/conv2d_72/Conv2D/ReadVariableOpReadVariableOp9decoder_block_18_conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
!decoder_block_18/conv2d_72/Conv2DConv2DReshape:output:08decoder_block_18/conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_18_conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
"decoder_block_18/conv2d_72/BiasAddBiasAdd*decoder_block_18/conv2d_72/Conv2D:output:09decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
decoder_block_18/conv2d_72/ReluRelu+decoder_block_18/conv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@x
'decoder_block_18/up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      z
)decoder_block_18/up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
%decoder_block_18/up_sampling2d_18/mulMul0decoder_block_18/up_sampling2d_18/Const:output:02decoder_block_18/up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
>decoder_block_18/up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighbor-decoder_block_18/conv2d_72/Relu:activations:0)decoder_block_18/up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
6decoder_block_18/batch_normalization_53/ReadVariableOpReadVariableOp?decoder_block_18_batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
8decoder_block_18/batch_normalization_53/ReadVariableOp_1ReadVariableOpAdecoder_block_18_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Gdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
8decoder_block_18/batch_normalization_53/FusedBatchNormV3FusedBatchNormV3Odecoder_block_18/up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0>decoder_block_18/batch_normalization_53/ReadVariableOp:value:0@decoder_block_18/batch_normalization_53/ReadVariableOp_1:value:0Odecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
6decoder_block_18/batch_normalization_53/AssignNewValueAssignVariableOpPdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_resourceEdecoder_block_18/batch_normalization_53/FusedBatchNormV3:batch_mean:0H^decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
8decoder_block_18/batch_normalization_53/AssignNewValue_1AssignVariableOpRdecoder_block_18_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resourceIdecoder_block_18/batch_normalization_53/FusedBatchNormV3:batch_variance:0J^decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
0decoder_block_19/conv2d_73/Conv2D/ReadVariableOpReadVariableOp9decoder_block_19_conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
!decoder_block_19/conv2d_73/Conv2DConv2D<decoder_block_18/batch_normalization_53/FusedBatchNormV3:y:08decoder_block_19/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_19_conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"decoder_block_19/conv2d_73/BiasAddBiasAdd*decoder_block_19/conv2d_73/Conv2D:output:09decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� �
decoder_block_19/conv2d_73/ReluRelu+decoder_block_19/conv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� x
'decoder_block_19/up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      z
)decoder_block_19/up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
%decoder_block_19/up_sampling2d_19/mulMul0decoder_block_19/up_sampling2d_19/Const:output:02decoder_block_19/up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
>decoder_block_19/up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighbor-decoder_block_19/conv2d_73/Relu:activations:0)decoder_block_19/up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
6decoder_block_19/batch_normalization_54/ReadVariableOpReadVariableOp?decoder_block_19_batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
8decoder_block_19/batch_normalization_54/ReadVariableOp_1ReadVariableOpAdecoder_block_19_batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Gdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
8decoder_block_19/batch_normalization_54/FusedBatchNormV3FusedBatchNormV3Odecoder_block_19/up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0>decoder_block_19/batch_normalization_54/ReadVariableOp:value:0@decoder_block_19/batch_normalization_54/ReadVariableOp_1:value:0Odecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
6decoder_block_19/batch_normalization_54/AssignNewValueAssignVariableOpPdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_resourceEdecoder_block_19/batch_normalization_54/FusedBatchNormV3:batch_mean:0H^decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
8decoder_block_19/batch_normalization_54/AssignNewValue_1AssignVariableOpRdecoder_block_19_batch_normalization_54_fusedbatchnormv3_readvariableop_1_resourceIdecoder_block_19/batch_normalization_54/FusedBatchNormV3:batch_variance:0J^decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
0decoder_block_20/conv2d_74/Conv2D/ReadVariableOpReadVariableOp9decoder_block_20_conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!decoder_block_20/conv2d_74/Conv2DConv2D<decoder_block_19/batch_normalization_54/FusedBatchNormV3:y:08decoder_block_20/conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_20_conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"decoder_block_20/conv2d_74/BiasAddBiasAdd*decoder_block_20/conv2d_74/Conv2D:output:09decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  �
decoder_block_20/conv2d_74/ReluRelu+decoder_block_20/conv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  x
'decoder_block_20/up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        z
)decoder_block_20/up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
%decoder_block_20/up_sampling2d_20/mulMul0decoder_block_20/up_sampling2d_20/Const:output:02decoder_block_20/up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
>decoder_block_20/up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighbor-decoder_block_20/conv2d_74/Relu:activations:0)decoder_block_20/up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
6decoder_block_20/batch_normalization_55/ReadVariableOpReadVariableOp?decoder_block_20_batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
8decoder_block_20/batch_normalization_55/ReadVariableOp_1ReadVariableOpAdecoder_block_20_batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Gdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
8decoder_block_20/batch_normalization_55/FusedBatchNormV3FusedBatchNormV3Odecoder_block_20/up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0>decoder_block_20/batch_normalization_55/ReadVariableOp:value:0@decoder_block_20/batch_normalization_55/ReadVariableOp_1:value:0Odecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
6decoder_block_20/batch_normalization_55/AssignNewValueAssignVariableOpPdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_resourceEdecoder_block_20/batch_normalization_55/FusedBatchNormV3:batch_mean:0H^decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
8decoder_block_20/batch_normalization_55/AssignNewValue_1AssignVariableOpRdecoder_block_20_batch_normalization_55_fusedbatchnormv3_readvariableop_1_resourceIdecoder_block_20/batch_normalization_55/FusedBatchNormV3:batch_variance:0J^decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_75/Conv2D/ReadVariableOpReadVariableOp(conv2d_75_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_75/Conv2DConv2D<decoder_block_20/batch_normalization_55/FusedBatchNormV3:y:0'conv2d_75/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_75/BiasAdd/ReadVariableOpReadVariableOp)conv2d_75_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_75/BiasAddBiasAddconv2d_75/Conv2D:output:0(conv2d_75/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������n
conv2d_75/ReluReluconv2d_75/BiasAdd:output:0*
T0*1
_output_shapes
:������������
conv2d_76/Conv2D/ReadVariableOpReadVariableOp(conv2d_76_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_76/Conv2DConv2Dconv2d_75/Relu:activations:0'conv2d_76/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
 conv2d_76/BiasAdd/ReadVariableOpReadVariableOp)conv2d_76_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_76/BiasAddBiasAddconv2d_76/Conv2D:output:0(conv2d_76/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������t
conv2d_76/SigmoidSigmoidconv2d_76/BiasAdd:output:0*
T0*1
_output_shapes
:�����������n
IdentityIdentityconv2d_76/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp!^conv2d_75/BiasAdd/ReadVariableOp ^conv2d_75/Conv2D/ReadVariableOp!^conv2d_76/BiasAdd/ReadVariableOp ^conv2d_76/Conv2D/ReadVariableOp7^decoder_block_18/batch_normalization_53/AssignNewValue9^decoder_block_18/batch_normalization_53/AssignNewValue_1H^decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpJ^decoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_17^decoder_block_18/batch_normalization_53/ReadVariableOp9^decoder_block_18/batch_normalization_53/ReadVariableOp_12^decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp1^decoder_block_18/conv2d_72/Conv2D/ReadVariableOp7^decoder_block_19/batch_normalization_54/AssignNewValue9^decoder_block_19/batch_normalization_54/AssignNewValue_1H^decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpJ^decoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_17^decoder_block_19/batch_normalization_54/ReadVariableOp9^decoder_block_19/batch_normalization_54/ReadVariableOp_12^decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp1^decoder_block_19/conv2d_73/Conv2D/ReadVariableOp7^decoder_block_20/batch_normalization_55/AssignNewValue9^decoder_block_20/batch_normalization_55/AssignNewValue_1H^decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpJ^decoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_17^decoder_block_20/batch_normalization_55/ReadVariableOp9^decoder_block_20/batch_normalization_55/ReadVariableOp_12^decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp1^decoder_block_20/conv2d_74/Conv2D/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_75/BiasAdd/ReadVariableOp conv2d_75/BiasAdd/ReadVariableOp2B
conv2d_75/Conv2D/ReadVariableOpconv2d_75/Conv2D/ReadVariableOp2D
 conv2d_76/BiasAdd/ReadVariableOp conv2d_76/BiasAdd/ReadVariableOp2B
conv2d_76/Conv2D/ReadVariableOpconv2d_76/Conv2D/ReadVariableOp2t
8decoder_block_18/batch_normalization_53/AssignNewValue_18decoder_block_18/batch_normalization_53/AssignNewValue_12p
6decoder_block_18/batch_normalization_53/AssignNewValue6decoder_block_18/batch_normalization_53/AssignNewValue2�
Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12�
Gdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOpGdecoder_block_18/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_18/batch_normalization_53/ReadVariableOp_18decoder_block_18/batch_normalization_53/ReadVariableOp_12p
6decoder_block_18/batch_normalization_53/ReadVariableOp6decoder_block_18/batch_normalization_53/ReadVariableOp2f
1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp1decoder_block_18/conv2d_72/BiasAdd/ReadVariableOp2d
0decoder_block_18/conv2d_72/Conv2D/ReadVariableOp0decoder_block_18/conv2d_72/Conv2D/ReadVariableOp2t
8decoder_block_19/batch_normalization_54/AssignNewValue_18decoder_block_19/batch_normalization_54/AssignNewValue_12p
6decoder_block_19/batch_normalization_54/AssignNewValue6decoder_block_19/batch_normalization_54/AssignNewValue2�
Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12�
Gdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOpGdecoder_block_19/batch_normalization_54/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_19/batch_normalization_54/ReadVariableOp_18decoder_block_19/batch_normalization_54/ReadVariableOp_12p
6decoder_block_19/batch_normalization_54/ReadVariableOp6decoder_block_19/batch_normalization_54/ReadVariableOp2f
1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp1decoder_block_19/conv2d_73/BiasAdd/ReadVariableOp2d
0decoder_block_19/conv2d_73/Conv2D/ReadVariableOp0decoder_block_19/conv2d_73/Conv2D/ReadVariableOp2t
8decoder_block_20/batch_normalization_55/AssignNewValue_18decoder_block_20/batch_normalization_55/AssignNewValue_12p
6decoder_block_20/batch_normalization_55/AssignNewValue6decoder_block_20/batch_normalization_55/AssignNewValue2�
Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12�
Gdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOpGdecoder_block_20/batch_normalization_55/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_20/batch_normalization_55/ReadVariableOp_18decoder_block_20/batch_normalization_55/ReadVariableOp_12p
6decoder_block_20/batch_normalization_55/ReadVariableOp6decoder_block_20/batch_normalization_55/ReadVariableOp2f
1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp1decoder_block_20/conv2d_74/BiasAdd/ReadVariableOp2d
0decoder_block_20/conv2d_74/Conv2D/ReadVariableOp0decoder_block_20/conv2d_74/Conv2D/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
�
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1916514

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918202

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1916704

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
2__inference_decoder_block_18_layer_call_fn_1917762
input_tensor!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916766w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�	
�
8__inference_batch_normalization_53_layer_call_fn_1918087

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1916348�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
N
2__inference_up_sampling2d_19_layer_call_fn_1918128

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
GPU2 *0J 8� *V
fQRO
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1916388�
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
�"
�
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916766
input_tensorB
(conv2d_72_conv2d_readvariableop_resource:@@7
)conv2d_72_biasadd_readvariableop_resource:@<
.batch_normalization_53_readvariableop_resource:@>
0batch_normalization_53_readvariableop_1_resource:@M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@
identity��6batch_normalization_53/FusedBatchNormV3/ReadVariableOp�8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_53/ReadVariableOp�'batch_normalization_53/ReadVariableOp_1� conv2d_72/BiasAdd/ReadVariableOp�conv2d_72/Conv2D/ReadVariableOp�
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_72/Conv2DConv2Dinput_tensor'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_18/mulMulup_sampling2d_18/Const:output:0!up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_72/Relu:activations:0up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_53/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp7^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�
�
%__inference_signature_wrapper_1917398
input_1
unknown:
�� 
	unknown_0:	� #
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@#
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

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
"__inference__wrapped_model_1916292y
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918281

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�"
�
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1918004
input_tensorB
(conv2d_74_conv2d_readvariableop_resource: 7
)conv2d_74_biasadd_readvariableop_resource:<
.batch_normalization_55_readvariableop_resource:>
0batch_normalization_55_readvariableop_1_resource:M
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:
identity��6batch_normalization_55/FusedBatchNormV3/ReadVariableOp�8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_55/ReadVariableOp�'batch_normalization_55/ReadVariableOp_1� conv2d_74/BiasAdd/ReadVariableOp�conv2d_74/Conv2D/ReadVariableOp�
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_74/Conv2DConv2Dinput_tensor'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  l
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  g
up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        i
up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_20/mulMulup_sampling2d_20/Const:output:0!up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_74/Relu:activations:0up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_55/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp7^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������   : : : : : : 2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������   
&
_user_specified_nameinput_tensor
�o
�
#__inference__traced_restore_1918530
file_prefix<
(assignvariableop_decoder_dense_20_kernel:
�� 7
(assignvariableop_1_decoder_dense_20_bias:	� V
<assignvariableop_2_decoder_decoder_block_18_conv2d_72_kernel:@@H
:assignvariableop_3_decoder_decoder_block_18_conv2d_72_bias:@V
Hassignvariableop_4_decoder_decoder_block_18_batch_normalization_53_gamma:@U
Gassignvariableop_5_decoder_decoder_block_18_batch_normalization_53_beta:@\
Nassignvariableop_6_decoder_decoder_block_18_batch_normalization_53_moving_mean:@`
Rassignvariableop_7_decoder_decoder_block_18_batch_normalization_53_moving_variance:@V
<assignvariableop_8_decoder_decoder_block_19_conv2d_73_kernel:@ H
:assignvariableop_9_decoder_decoder_block_19_conv2d_73_bias: W
Iassignvariableop_10_decoder_decoder_block_19_batch_normalization_54_gamma: V
Hassignvariableop_11_decoder_decoder_block_19_batch_normalization_54_beta: ]
Oassignvariableop_12_decoder_decoder_block_19_batch_normalization_54_moving_mean: a
Sassignvariableop_13_decoder_decoder_block_19_batch_normalization_54_moving_variance: W
=assignvariableop_14_decoder_decoder_block_20_conv2d_74_kernel: I
;assignvariableop_15_decoder_decoder_block_20_conv2d_74_bias:W
Iassignvariableop_16_decoder_decoder_block_20_batch_normalization_55_gamma:V
Hassignvariableop_17_decoder_decoder_block_20_batch_normalization_55_beta:]
Oassignvariableop_18_decoder_decoder_block_20_batch_normalization_55_moving_mean:a
Sassignvariableop_19_decoder_decoder_block_20_batch_normalization_55_moving_variance:F
,assignvariableop_20_decoder_conv2d_75_kernel:8
*assignvariableop_21_decoder_conv2d_75_bias:F
,assignvariableop_22_decoder_conv2d_76_kernel:8
*assignvariableop_23_decoder_conv2d_76_bias:
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
AssignVariableOpAssignVariableOp(assignvariableop_decoder_dense_20_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_decoder_dense_20_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp<assignvariableop_2_decoder_decoder_block_18_conv2d_72_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp:assignvariableop_3_decoder_decoder_block_18_conv2d_72_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpHassignvariableop_4_decoder_decoder_block_18_batch_normalization_53_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpGassignvariableop_5_decoder_decoder_block_18_batch_normalization_53_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpNassignvariableop_6_decoder_decoder_block_18_batch_normalization_53_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpRassignvariableop_7_decoder_decoder_block_18_batch_normalization_53_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp<assignvariableop_8_decoder_decoder_block_19_conv2d_73_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp:assignvariableop_9_decoder_decoder_block_19_conv2d_73_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpIassignvariableop_10_decoder_decoder_block_19_batch_normalization_54_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpHassignvariableop_11_decoder_decoder_block_19_batch_normalization_54_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpOassignvariableop_12_decoder_decoder_block_19_batch_normalization_54_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpSassignvariableop_13_decoder_decoder_block_19_batch_normalization_54_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp=assignvariableop_14_decoder_decoder_block_20_conv2d_74_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp;assignvariableop_15_decoder_decoder_block_20_conv2d_74_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpIassignvariableop_16_decoder_decoder_block_20_batch_normalization_55_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpHassignvariableop_17_decoder_decoder_block_20_batch_normalization_55_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpOassignvariableop_18_decoder_decoder_block_20_batch_normalization_55_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpSassignvariableop_19_decoder_decoder_block_20_batch_normalization_55_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_decoder_conv2d_75_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp*assignvariableop_21_decoder_conv2d_75_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_decoder_conv2d_76_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_decoder_conv2d_76_biasIdentity_23:output:0"/device:CPU:0*&
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
2__inference_decoder_block_19_layer_call_fn_1917854
input_tensor!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916808w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�*
�
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916593
input_tensorB
(conv2d_72_conv2d_readvariableop_resource:@@7
)conv2d_72_biasadd_readvariableop_resource:@<
.batch_normalization_53_readvariableop_resource:@>
0batch_normalization_53_readvariableop_1_resource:@M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@
identity��%batch_normalization_53/AssignNewValue�'batch_normalization_53/AssignNewValue_1�6batch_normalization_53/FusedBatchNormV3/ReadVariableOp�8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_53/ReadVariableOp�'batch_normalization_53/ReadVariableOp_1� conv2d_72/BiasAdd/ReadVariableOp�conv2d_72/Conv2D/ReadVariableOp�
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_72/Conv2DConv2Dinput_tensor'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_18/mulMulup_sampling2d_18/Const:output:0!up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_72/Relu:activations:0up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_53/AssignNewValueAssignVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource4batch_normalization_53/FusedBatchNormV3:batch_mean:07^batch_normalization_53/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_53/AssignNewValue_1AssignVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_53/FusedBatchNormV3:batch_variance:09^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_53/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp&^batch_normalization_53/AssignNewValue(^batch_normalization_53/AssignNewValue_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2R
'batch_normalization_53/AssignNewValue_1'batch_normalization_53/AssignNewValue_12N
%batch_normalization_53/AssignNewValue%batch_normalization_53/AssignNewValue2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�	
�
2__inference_decoder_block_19_layer_call_fn_1917837
input_tensor!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916636w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������   `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�*
�
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916636
input_tensorB
(conv2d_73_conv2d_readvariableop_resource:@ 7
)conv2d_73_biasadd_readvariableop_resource: <
.batch_normalization_54_readvariableop_resource: >
0batch_normalization_54_readvariableop_1_resource: M
?batch_normalization_54_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: 
identity��%batch_normalization_54/AssignNewValue�'batch_normalization_54/AssignNewValue_1�6batch_normalization_54/FusedBatchNormV3/ReadVariableOp�8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_54/ReadVariableOp�'batch_normalization_54/ReadVariableOp_1� conv2d_73/BiasAdd/ReadVariableOp�conv2d_73/Conv2D/ReadVariableOp�
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_73/Conv2DConv2Dinput_tensor'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� l
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� g
up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_19/mulMulup_sampling2d_19/Const:output:0!up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_73/Relu:activations:0up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_54/AssignNewValueAssignVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource4batch_normalization_54/FusedBatchNormV3:batch_mean:07^batch_normalization_54/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_54/AssignNewValue_1AssignVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_54/FusedBatchNormV3:batch_variance:09^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_54/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������   �
NoOpNoOp&^batch_normalization_54/AssignNewValue(^batch_normalization_54/AssignNewValue_17^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_1!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2R
'batch_normalization_54/AssignNewValue_1'batch_normalization_54/AssignNewValue_12N
%batch_normalization_54/AssignNewValue%batch_normalization_54/AssignNewValue2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�,
�

D__inference_decoder_layer_call_and_return_conditional_losses_1916728
input_1$
dense_20_1916557:
�� 
dense_20_1916559:	� 2
decoder_block_18_1916594:@@&
decoder_block_18_1916596:@&
decoder_block_18_1916598:@&
decoder_block_18_1916600:@&
decoder_block_18_1916602:@&
decoder_block_18_1916604:@2
decoder_block_19_1916637:@ &
decoder_block_19_1916639: &
decoder_block_19_1916641: &
decoder_block_19_1916643: &
decoder_block_19_1916645: &
decoder_block_19_1916647: 2
decoder_block_20_1916680: &
decoder_block_20_1916682:&
decoder_block_20_1916684:&
decoder_block_20_1916686:&
decoder_block_20_1916688:&
decoder_block_20_1916690:+
conv2d_75_1916705:
conv2d_75_1916707:+
conv2d_76_1916722:
conv2d_76_1916724:
identity��!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�(decoder_block_18/StatefulPartitionedCall�(decoder_block_19/StatefulPartitionedCall�(decoder_block_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_20_1916557dense_20_1916559*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1916556f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
ReshapeReshape)dense_20/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_18_1916594decoder_block_18_1916596decoder_block_18_1916598decoder_block_18_1916600decoder_block_18_1916602decoder_block_18_1916604*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916593�
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0decoder_block_19_1916637decoder_block_19_1916639decoder_block_19_1916641decoder_block_19_1916643decoder_block_19_1916645decoder_block_19_1916647*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916636�
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0decoder_block_20_1916680decoder_block_20_1916682decoder_block_20_1916684decoder_block_20_1916686decoder_block_20_1916688decoder_block_20_1916690*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916679�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_75_1916705conv2d_75_1916707*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1916704�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0conv2d_76_1916722conv2d_76_1916724*
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1916721�
IdentityIdentity*conv2d_76/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
*__inference_dense_20_layer_call_fn_1917717

inputs
unknown:
�� 
	unknown_0:	� 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1916556p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:���������� `
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
�*
�
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917791
input_tensorB
(conv2d_72_conv2d_readvariableop_resource:@@7
)conv2d_72_biasadd_readvariableop_resource:@<
.batch_normalization_53_readvariableop_resource:@>
0batch_normalization_53_readvariableop_1_resource:@M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:@
identity��%batch_normalization_53/AssignNewValue�'batch_normalization_53/AssignNewValue_1�6batch_normalization_53/FusedBatchNormV3/ReadVariableOp�8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_53/ReadVariableOp�'batch_normalization_53/ReadVariableOp_1� conv2d_72/BiasAdd/ReadVariableOp�conv2d_72/Conv2D/ReadVariableOp�
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
conv2d_72/Conv2DConv2Dinput_tensor'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@l
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*/
_output_shapes
:���������@g
up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_18/mulMulup_sampling2d_18/Const:output:0!up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_72/Relu:activations:0up_sampling2d_18/mul:z:0*
T0*/
_output_shapes
:���������@*
half_pixel_centers(�
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_53/AssignNewValueAssignVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource4batch_normalization_53/FusedBatchNormV3:batch_mean:07^batch_normalization_53/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_53/AssignNewValue_1AssignVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_53/FusedBatchNormV3:batch_variance:09^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_53/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������@�
NoOpNoOp&^batch_normalization_53/AssignNewValue(^batch_normalization_53/AssignNewValue_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2R
'batch_normalization_53/AssignNewValue_1'batch_normalization_53/AssignNewValue_12N
%batch_normalization_53/AssignNewValue%batch_normalization_53/AssignNewValue2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�"
�
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917912
input_tensorB
(conv2d_73_conv2d_readvariableop_resource:@ 7
)conv2d_73_biasadd_readvariableop_resource: <
.batch_normalization_54_readvariableop_resource: >
0batch_normalization_54_readvariableop_1_resource: M
?batch_normalization_54_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource: 
identity��6batch_normalization_54/FusedBatchNormV3/ReadVariableOp�8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_54/ReadVariableOp�'batch_normalization_54/ReadVariableOp_1� conv2d_73/BiasAdd/ReadVariableOp�conv2d_73/Conv2D/ReadVariableOp�
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_73/Conv2DConv2Dinput_tensor'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
�
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� l
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:��������� g
up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_19/mulMulup_sampling2d_19/Const:output:0!up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_73/Relu:activations:0up_sampling2d_19/mul:z:0*
T0*/
_output_shapes
:���������   *
half_pixel_centers(�
%batch_normalization_54/ReadVariableOpReadVariableOp.batch_normalization_54_readvariableop_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/ReadVariableOp_1ReadVariableOp0batch_normalization_54_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6batch_normalization_54/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_54_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_54_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
'batch_normalization_54/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_54/ReadVariableOp:value:0/batch_normalization_54/ReadVariableOp_1:value:0>batch_normalization_54/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������   : : : : :*
epsilon%o�:*
is_training( �
IdentityIdentity+batch_normalization_54/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:���������   �
NoOpNoOp7^batch_normalization_54/FusedBatchNormV3/ReadVariableOp9^batch_normalization_54/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_54/ReadVariableOp(^batch_normalization_54/ReadVariableOp_1!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������@: : : : : : 2t
8batch_normalization_54/FusedBatchNormV3/ReadVariableOp_18batch_normalization_54/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_54/FusedBatchNormV3/ReadVariableOp6batch_normalization_54/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_54/ReadVariableOp_1'batch_normalization_54/ReadVariableOp_12N
%batch_normalization_54/ReadVariableOp%batch_normalization_54/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������@
&
_user_specified_nameinput_tensor
�	
�
8__inference_batch_normalization_55_layer_call_fn_1918232

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *\
fWRU
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1916496�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1916413

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
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
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1917102
input_1
unknown:
�� 
	unknown_0:	� #
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@#
	unknown_7:@ 
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:

unknown_18:$

unknown_19:

unknown_20:$

unknown_21:

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
D__inference_decoder_layer_call_and_return_conditional_losses_1917051y
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
�

�
E__inference_dense_20_layer_call_and_return_conditional_losses_1916556

inputs2
matmul_readvariableop_resource:
�� .
biasadd_readvariableop_resource:	� 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�� *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:� *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:���������� Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:���������� b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:���������� w
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
�,
�

D__inference_decoder_layer_call_and_return_conditional_losses_1916875
input_1$
dense_20_1916731:
�� 
dense_20_1916733:	� 2
decoder_block_18_1916767:@@&
decoder_block_18_1916769:@&
decoder_block_18_1916771:@&
decoder_block_18_1916773:@&
decoder_block_18_1916775:@&
decoder_block_18_1916777:@2
decoder_block_19_1916809:@ &
decoder_block_19_1916811: &
decoder_block_19_1916813: &
decoder_block_19_1916815: &
decoder_block_19_1916817: &
decoder_block_19_1916819: 2
decoder_block_20_1916851: &
decoder_block_20_1916853:&
decoder_block_20_1916855:&
decoder_block_20_1916857:&
decoder_block_20_1916859:&
decoder_block_20_1916861:+
conv2d_75_1916864:
conv2d_75_1916866:+
conv2d_76_1916869:
conv2d_76_1916871:
identity��!conv2d_75/StatefulPartitionedCall�!conv2d_76/StatefulPartitionedCall�(decoder_block_18/StatefulPartitionedCall�(decoder_block_19/StatefulPartitionedCall�(decoder_block_20/StatefulPartitionedCall� dense_20/StatefulPartitionedCall�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_20_1916731dense_20_1916733*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:���������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1916556f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����      @   �
ReshapeReshape)dense_20/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*/
_output_shapes
:���������@�
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_18_1916767decoder_block_18_1916769decoder_block_18_1916771decoder_block_18_1916773decoder_block_18_1916775decoder_block_18_1916777*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1916766�
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0decoder_block_19_1916809decoder_block_19_1916811decoder_block_19_1916813decoder_block_19_1916815decoder_block_19_1916817decoder_block_19_1916819*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������   *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1916808�
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0decoder_block_20_1916851decoder_block_20_1916853decoder_block_20_1916855decoder_block_20_1916857decoder_block_20_1916859decoder_block_20_1916861*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1916850�
!conv2d_75/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_75_1916864conv2d_75_1916866*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *O
fJRH
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1916704�
!conv2d_76/StatefulPartitionedCallStatefulPartitionedCall*conv2d_75/StatefulPartitionedCall:output:0conv2d_76_1916869conv2d_76_1916871*
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1916721�
IdentityIdentity*conv2d_76/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp"^conv2d_75/StatefulPartitionedCall"^conv2d_76/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_20/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_75/StatefulPartitionedCall!conv2d_75/StatefulPartitionedCall2F
!conv2d_76/StatefulPartitionedCall!conv2d_76/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�*
�
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1917975
input_tensorB
(conv2d_74_conv2d_readvariableop_resource: 7
)conv2d_74_biasadd_readvariableop_resource:<
.batch_normalization_55_readvariableop_resource:>
0batch_normalization_55_readvariableop_1_resource:M
?batch_normalization_55_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource:
identity��%batch_normalization_55/AssignNewValue�'batch_normalization_55/AssignNewValue_1�6batch_normalization_55/FusedBatchNormV3/ReadVariableOp�8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1�%batch_normalization_55/ReadVariableOp�'batch_normalization_55/ReadVariableOp_1� conv2d_74/BiasAdd/ReadVariableOp�conv2d_74/Conv2D/ReadVariableOp�
conv2d_74/Conv2D/ReadVariableOpReadVariableOp(conv2d_74_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_74/Conv2DConv2Dinput_tensor'conv2d_74/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingSAME*
strides
�
 conv2d_74/BiasAdd/ReadVariableOpReadVariableOp)conv2d_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_74/BiasAddBiasAddconv2d_74/Conv2D:output:0(conv2d_74/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  l
conv2d_74/ReluReluconv2d_74/BiasAdd:output:0*
T0*/
_output_shapes
:���������  g
up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        i
up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_20/mulMulup_sampling2d_20/Const:output:0!up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:�
-up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_74/Relu:activations:0up_sampling2d_20/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(�
%batch_normalization_55/ReadVariableOpReadVariableOp.batch_normalization_55_readvariableop_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/ReadVariableOp_1ReadVariableOp0batch_normalization_55_readvariableop_1_resource*
_output_shapes
:*
dtype0�
6batch_normalization_55/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'batch_normalization_55/FusedBatchNormV3FusedBatchNormV3>up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0-batch_normalization_55/ReadVariableOp:value:0/batch_normalization_55/ReadVariableOp_1:value:0>batch_normalization_55/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:�����������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
%batch_normalization_55/AssignNewValueAssignVariableOp?batch_normalization_55_fusedbatchnormv3_readvariableop_resource4batch_normalization_55/FusedBatchNormV3:batch_mean:07^batch_normalization_55/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
'batch_normalization_55/AssignNewValue_1AssignVariableOpAbatch_normalization_55_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_55/FusedBatchNormV3:batch_variance:09^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity+batch_normalization_55/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp&^batch_normalization_55/AssignNewValue(^batch_normalization_55/AssignNewValue_17^batch_normalization_55/FusedBatchNormV3/ReadVariableOp9^batch_normalization_55/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_55/ReadVariableOp(^batch_normalization_55/ReadVariableOp_1!^conv2d_74/BiasAdd/ReadVariableOp ^conv2d_74/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������   : : : : : : 2R
'batch_normalization_55/AssignNewValue_1'batch_normalization_55/AssignNewValue_12N
%batch_normalization_55/AssignNewValue%batch_normalization_55/AssignNewValue2t
8batch_normalization_55/FusedBatchNormV3/ReadVariableOp_18batch_normalization_55/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_55/FusedBatchNormV3/ReadVariableOp6batch_normalization_55/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_55/ReadVariableOp_1'batch_normalization_55/ReadVariableOp_12N
%batch_normalization_55/ReadVariableOp%batch_normalization_55/ReadVariableOp2D
 conv2d_74/BiasAdd/ReadVariableOp conv2d_74/BiasAdd/ReadVariableOp2B
conv2d_74/Conv2D/ReadVariableOpconv2d_74/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:���������   
&
_user_specified_nameinput_tensor
�
�
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1916348

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
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
)__inference_decoder_layer_call_fn_1916989
)__inference_decoder_layer_call_fn_1917102
)__inference_decoder_layer_call_fn_1917451
)__inference_decoder_layer_call_fn_1917504�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1916728
D__inference_decoder_layer_call_and_return_conditional_losses_1916875
D__inference_decoder_layer_call_and_return_conditional_losses_1917606
D__inference_decoder_layer_call_and_return_conditional_losses_1917708�
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
"__inference__wrapped_model_1916292input_1"�
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
+:)
�� 2decoder/dense_20/kernel
$:"� 2decoder/dense_20/bias
C:A@@2)decoder/decoder_block_18/conv2d_72/kernel
5:3@2'decoder/decoder_block_18/conv2d_72/bias
C:A@25decoder/decoder_block_18/batch_normalization_53/gamma
B:@@24decoder/decoder_block_18/batch_normalization_53/beta
K:I@ (2;decoder/decoder_block_18/batch_normalization_53/moving_mean
O:M@ (2?decoder/decoder_block_18/batch_normalization_53/moving_variance
C:A@ 2)decoder/decoder_block_19/conv2d_73/kernel
5:3 2'decoder/decoder_block_19/conv2d_73/bias
C:A 25decoder/decoder_block_19/batch_normalization_54/gamma
B:@ 24decoder/decoder_block_19/batch_normalization_54/beta
K:I  (2;decoder/decoder_block_19/batch_normalization_54/moving_mean
O:M  (2?decoder/decoder_block_19/batch_normalization_54/moving_variance
C:A 2)decoder/decoder_block_20/conv2d_74/kernel
5:32'decoder/decoder_block_20/conv2d_74/bias
C:A25decoder/decoder_block_20/batch_normalization_55/gamma
B:@24decoder/decoder_block_20/batch_normalization_55/beta
K:I (2;decoder/decoder_block_20/batch_normalization_55/moving_mean
O:M (2?decoder/decoder_block_20/batch_normalization_55/moving_variance
2:02decoder/conv2d_75/kernel
$:"2decoder/conv2d_75/bias
2:02decoder/conv2d_76/kernel
$:"2decoder/conv2d_76/bias
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
)__inference_decoder_layer_call_fn_1916989input_1"�
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
)__inference_decoder_layer_call_fn_1917102input_1"�
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
)__inference_decoder_layer_call_fn_1917451embedding_input"�
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
)__inference_decoder_layer_call_fn_1917504embedding_input"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1916728input_1"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1916875input_1"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1917606embedding_input"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1917708embedding_input"�
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
*__inference_dense_20_layer_call_fn_1917717�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_1917728�
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
2__inference_decoder_block_18_layer_call_fn_1917745
2__inference_decoder_block_18_layer_call_fn_1917762�
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
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917791
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917820�
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
2__inference_decoder_block_19_layer_call_fn_1917837
2__inference_decoder_block_19_layer_call_fn_1917854�
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
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917883
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917912�
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
2__inference_decoder_block_20_layer_call_fn_1917929
2__inference_decoder_block_20_layer_call_fn_1917946�
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
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1917975
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1918004�
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
+__inference_conv2d_75_layer_call_fn_1918013�
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
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1918024�
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
+__inference_conv2d_76_layer_call_fn_1918033�
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1918044�
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
%__inference_signature_wrapper_1917398input_1"�
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
*__inference_dense_20_layer_call_fn_1917717inputs"�
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
E__inference_dense_20_layer_call_and_return_conditional_losses_1917728inputs"�
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
2__inference_decoder_block_18_layer_call_fn_1917745input_tensor"�
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
2__inference_decoder_block_18_layer_call_fn_1917762input_tensor"�
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
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917791input_tensor"�
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
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917820input_tensor"�
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
2__inference_up_sampling2d_18_layer_call_fn_1918049�
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
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1918061�
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
8__inference_batch_normalization_53_layer_call_fn_1918074
8__inference_batch_normalization_53_layer_call_fn_1918087�
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918105
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918123�
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
2__inference_decoder_block_19_layer_call_fn_1917837input_tensor"�
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
2__inference_decoder_block_19_layer_call_fn_1917854input_tensor"�
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
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917883input_tensor"�
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
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917912input_tensor"�
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
2__inference_up_sampling2d_19_layer_call_fn_1918128�
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
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1918140�
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
8__inference_batch_normalization_54_layer_call_fn_1918153
8__inference_batch_normalization_54_layer_call_fn_1918166�
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918184
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918202�
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
2__inference_decoder_block_20_layer_call_fn_1917929input_tensor"�
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
2__inference_decoder_block_20_layer_call_fn_1917946input_tensor"�
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
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1917975input_tensor"�
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
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1918004input_tensor"�
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
2__inference_up_sampling2d_20_layer_call_fn_1918207�
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
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1918219�
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
8__inference_batch_normalization_55_layer_call_fn_1918232
8__inference_batch_normalization_55_layer_call_fn_1918245�
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918263
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918281�
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
+__inference_conv2d_75_layer_call_fn_1918013inputs"�
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
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1918024inputs"�
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
+__inference_conv2d_76_layer_call_fn_1918033inputs"�
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
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1918044inputs"�
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
2__inference_up_sampling2d_18_layer_call_fn_1918049inputs"�
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
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1918061inputs"�
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
8__inference_batch_normalization_53_layer_call_fn_1918074inputs"�
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
8__inference_batch_normalization_53_layer_call_fn_1918087inputs"�
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918105inputs"�
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
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918123inputs"�
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
2__inference_up_sampling2d_19_layer_call_fn_1918128inputs"�
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
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1918140inputs"�
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
8__inference_batch_normalization_54_layer_call_fn_1918153inputs"�
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
8__inference_batch_normalization_54_layer_call_fn_1918166inputs"�
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918184inputs"�
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
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918202inputs"�
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
2__inference_up_sampling2d_20_layer_call_fn_1918207inputs"�
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
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1918219inputs"�
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
8__inference_batch_normalization_55_layer_call_fn_1918232inputs"�
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
8__inference_batch_normalization_55_layer_call_fn_1918245inputs"�
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918263inputs"�
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
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918281inputs"�
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
"__inference__wrapped_model_1916292� !"#$%&1�.
'�$
"�
input_1����������
� "=�:
8
output_1,�)
output_1������������
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918105�Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
S__inference_batch_normalization_53_layer_call_and_return_conditional_losses_1918123�Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
8__inference_batch_normalization_53_layer_call_fn_1918074�Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
8__inference_batch_normalization_53_layer_call_fn_1918087�Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918184�Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
S__inference_batch_normalization_54_layer_call_and_return_conditional_losses_1918202�Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
8__inference_batch_normalization_54_layer_call_fn_1918153�Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
8__inference_batch_normalization_54_layer_call_fn_1918166�Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918263� !"Q�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
S__inference_batch_normalization_55_layer_call_and_return_conditional_losses_1918281� !"Q�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
8__inference_batch_normalization_55_layer_call_fn_1918232� !"Q�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
8__inference_batch_normalization_55_layer_call_fn_1918245� !"Q�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
F__inference_conv2d_75_layer_call_and_return_conditional_losses_1918024w#$9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
+__inference_conv2d_75_layer_call_fn_1918013l#$9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
F__inference_conv2d_76_layer_call_and_return_conditional_losses_1918044w%&9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
+__inference_conv2d_76_layer_call_fn_1918033l%&9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917791�A�>
7�4
.�+
input_tensor���������@
p
� "4�1
*�'
tensor_0���������@
� �
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1917820�A�>
7�4
.�+
input_tensor���������@
p 
� "4�1
*�'
tensor_0���������@
� �
2__inference_decoder_block_18_layer_call_fn_1917745vA�>
7�4
.�+
input_tensor���������@
p
� ")�&
unknown���������@�
2__inference_decoder_block_18_layer_call_fn_1917762vA�>
7�4
.�+
input_tensor���������@
p 
� ")�&
unknown���������@�
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917883�A�>
7�4
.�+
input_tensor���������@
p
� "4�1
*�'
tensor_0���������   
� �
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1917912�A�>
7�4
.�+
input_tensor���������@
p 
� "4�1
*�'
tensor_0���������   
� �
2__inference_decoder_block_19_layer_call_fn_1917837vA�>
7�4
.�+
input_tensor���������@
p
� ")�&
unknown���������   �
2__inference_decoder_block_19_layer_call_fn_1917854vA�>
7�4
.�+
input_tensor���������@
p 
� ")�&
unknown���������   �
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1917975� !"A�>
7�4
.�+
input_tensor���������   
p
� "6�3
,�)
tensor_0�����������
� �
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1918004� !"A�>
7�4
.�+
input_tensor���������   
p 
� "6�3
,�)
tensor_0�����������
� �
2__inference_decoder_block_20_layer_call_fn_1917929x !"A�>
7�4
.�+
input_tensor���������   
p
� "+�(
unknown������������
2__inference_decoder_block_20_layer_call_fn_1917946x !"A�>
7�4
.�+
input_tensor���������   
p 
� "+�(
unknown������������
D__inference_decoder_layer_call_and_return_conditional_losses_1916728� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1916875� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1917606� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1917708� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
)__inference_decoder_layer_call_fn_1916989� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp"+�(
unknown������������
)__inference_decoder_layer_call_fn_1917102� !"#$%&A�>
'�$
"�
input_1����������
�

trainingp "+�(
unknown������������
)__inference_decoder_layer_call_fn_1917451� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp"+�(
unknown������������
)__inference_decoder_layer_call_fn_1917504� !"#$%&I�F
/�,
*�'
embedding_input����������
�

trainingp "+�(
unknown������������
E__inference_dense_20_layer_call_and_return_conditional_losses_1917728e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0���������� 
� �
*__inference_dense_20_layer_call_fn_1917717Z0�-
&�#
!�
inputs����������
� ""�
unknown���������� �
%__inference_signature_wrapper_1917398� !"#$%&<�9
� 
2�/
-
input_1"�
input_1����������"=�:
8
output_1,�)
output_1������������
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1918061�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_up_sampling2d_18_layer_call_fn_1918049�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1918140�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_up_sampling2d_19_layer_call_fn_1918128�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1918219�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
2__inference_up_sampling2d_20_layer_call_fn_1918207�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4������������������������������������