��
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
decoder/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedecoder/conv2d_8/bias
{
)decoder/conv2d_8/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_8/bias*
_output_shapes
:*
dtype0
�
decoder/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namedecoder/conv2d_8/kernel
�
+decoder/conv2d_8/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_8/kernel*&
_output_shapes
:*
dtype0
�
decoder/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedecoder/conv2d_7/bias
{
)decoder/conv2d_7/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_7/bias*
_output_shapes
:*
dtype0
�
decoder/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedecoder/conv2d_7/kernel
�
+decoder/conv2d_7/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_7/kernel*&
_output_shapes
: *
dtype0
�
=decoder/decoder_block_2/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=decoder/decoder_block_2/batch_normalization_6/moving_variance
�
Qdecoder/decoder_block_2/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp=decoder/decoder_block_2/batch_normalization_6/moving_variance*
_output_shapes
: *
dtype0
�
9decoder/decoder_block_2/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9decoder/decoder_block_2/batch_normalization_6/moving_mean
�
Mdecoder/decoder_block_2/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp9decoder/decoder_block_2/batch_normalization_6/moving_mean*
_output_shapes
: *
dtype0
�
2decoder/decoder_block_2/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42decoder/decoder_block_2/batch_normalization_6/beta
�
Fdecoder/decoder_block_2/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp2decoder/decoder_block_2/batch_normalization_6/beta*
_output_shapes
: *
dtype0
�
3decoder/decoder_block_2/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53decoder/decoder_block_2/batch_normalization_6/gamma
�
Gdecoder/decoder_block_2/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_2/batch_normalization_6/gamma*
_output_shapes
: *
dtype0
�
%decoder/decoder_block_2/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%decoder/decoder_block_2/conv2d_6/bias
�
9decoder/decoder_block_2/conv2d_6/bias/Read/ReadVariableOpReadVariableOp%decoder/decoder_block_2/conv2d_6/bias*
_output_shapes
: *
dtype0
�
'decoder/decoder_block_2/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *8
shared_name)'decoder/decoder_block_2/conv2d_6/kernel
�
;decoder/decoder_block_2/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_2/conv2d_6/kernel*&
_output_shapes
:@ *
dtype0
�
=decoder/decoder_block_1/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*N
shared_name?=decoder/decoder_block_1/batch_normalization_5/moving_variance
�
Qdecoder/decoder_block_1/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp=decoder/decoder_block_1/batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
�
9decoder/decoder_block_1/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*J
shared_name;9decoder/decoder_block_1/batch_normalization_5/moving_mean
�
Mdecoder/decoder_block_1/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp9decoder/decoder_block_1/batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
�
2decoder/decoder_block_1/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42decoder/decoder_block_1/batch_normalization_5/beta
�
Fdecoder/decoder_block_1/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp2decoder/decoder_block_1/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
�
3decoder/decoder_block_1/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53decoder/decoder_block_1/batch_normalization_5/gamma
�
Gdecoder/decoder_block_1/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_1/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0
�
%decoder/decoder_block_1/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%decoder/decoder_block_1/conv2d_5/bias
�
9decoder/decoder_block_1/conv2d_5/bias/Read/ReadVariableOpReadVariableOp%decoder/decoder_block_1/conv2d_5/bias*
_output_shapes
:@*
dtype0
�
'decoder/decoder_block_1/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:�@*8
shared_name)'decoder/decoder_block_1/conv2d_5/kernel
�
;decoder/decoder_block_1/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_1/conv2d_5/kernel*'
_output_shapes
:�@*
dtype0
�
;decoder/decoder_block/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*L
shared_name=;decoder/decoder_block/batch_normalization_4/moving_variance
�
Odecoder/decoder_block/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp;decoder/decoder_block/batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
7decoder/decoder_block/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*H
shared_name97decoder/decoder_block/batch_normalization_4/moving_mean
�
Kdecoder/decoder_block/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp7decoder/decoder_block/batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
0decoder/decoder_block/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*A
shared_name20decoder/decoder_block/batch_normalization_4/beta
�
Ddecoder/decoder_block/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp0decoder/decoder_block/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
1decoder/decoder_block/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*B
shared_name31decoder/decoder_block/batch_normalization_4/gamma
�
Edecoder/decoder_block/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp1decoder/decoder_block/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
#decoder/decoder_block/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#decoder/decoder_block/conv2d_4/bias
�
7decoder/decoder_block/conv2d_4/bias/Read/ReadVariableOpReadVariableOp#decoder/decoder_block/conv2d_4/bias*
_output_shapes	
:�*
dtype0
�
%decoder/decoder_block/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*6
shared_name'%decoder/decoder_block/conv2d_4/kernel
�
9decoder/decoder_block/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp%decoder/decoder_block/conv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
decoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*%
shared_namedecoder/dense_2/bias
{
(decoder/dense_2/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_2/bias*
_output_shapes

:��*
dtype0
�
decoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*'
shared_namedecoder/dense_2/kernel
�
*decoder/dense_2/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_2/kernel*!
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1decoder/dense_2/kerneldecoder/dense_2/bias%decoder/decoder_block/conv2d_4/kernel#decoder/decoder_block/conv2d_4/bias1decoder/decoder_block/batch_normalization_4/gamma0decoder/decoder_block/batch_normalization_4/beta7decoder/decoder_block/batch_normalization_4/moving_mean;decoder/decoder_block/batch_normalization_4/moving_variance'decoder/decoder_block_1/conv2d_5/kernel%decoder/decoder_block_1/conv2d_5/bias3decoder/decoder_block_1/batch_normalization_5/gamma2decoder/decoder_block_1/batch_normalization_5/beta9decoder/decoder_block_1/batch_normalization_5/moving_mean=decoder/decoder_block_1/batch_normalization_5/moving_variance'decoder/decoder_block_2/conv2d_6/kernel%decoder/decoder_block_2/conv2d_6/bias3decoder/decoder_block_2/batch_normalization_6/gamma2decoder/decoder_block_2/batch_normalization_6/beta9decoder/decoder_block_2/batch_normalization_6/moving_mean=decoder/decoder_block_2/batch_normalization_6/moving_variancedecoder/conv2d_7/kerneldecoder/conv2d_7/biasdecoder/conv2d_8/kerneldecoder/conv2d_8/bias*$
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
%__inference_signature_wrapper_1974432

NoOpNoOp
�e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�d
value�dB�d B�d
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
firts_layer

	upsam1


block1

upsam2

block2

upsam3

block3
conv
	final

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17
$18
%19
&20
'21
(22
)23*
�
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
&14
'15
(16
)17*
* 
�
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
/trace_0
0trace_1
1trace_2
2trace_3* 
6
3trace_0
4trace_1
5trace_2
6trace_3* 
* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

kernel
bias*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jbn*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Wconv
Xbn*
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
econv
fbn*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

&kernel
'bias
 m_jit_compiled_convolution_op*
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

(kernel
)bias
 t_jit_compiled_convolution_op*

userving_default* 
VP
VARIABLE_VALUEdecoder/dense_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdecoder/dense_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%decoder/decoder_block/conv2d_4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#decoder/decoder_block/conv2d_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1decoder/decoder_block/batch_normalization_4/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0decoder/decoder_block/batch_normalization_4/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE7decoder/decoder_block/batch_normalization_4/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;decoder/decoder_block/batch_normalization_4/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'decoder/decoder_block_1/conv2d_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%decoder/decoder_block_1/conv2d_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3decoder/decoder_block_1/batch_normalization_5/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2decoder/decoder_block_1/batch_normalization_5/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9decoder/decoder_block_1/batch_normalization_5/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=decoder/decoder_block_1/batch_normalization_5/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'decoder/decoder_block_2/conv2d_6/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%decoder/decoder_block_2/conv2d_6/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3decoder/decoder_block_2/batch_normalization_6/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2decoder/decoder_block_2/batch_normalization_6/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9decoder/decoder_block_2/batch_normalization_6/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=decoder/decoder_block_2/batch_normalization_6/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEdecoder/conv2d_7/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdecoder/conv2d_7/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEdecoder/conv2d_8/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdecoder/conv2d_8/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
2
3
$4
%5*
C
0
	1

2
3
4
5
6
7
8*
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
0
1*

0
1*
* 
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

{trace_0* 

|trace_0* 
* 
* 
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
.
0
1
2
3
4
5*
 
0
1
2
3*
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

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

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

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
.
 0
!1
"2
#3
$4
%5*
 
 0
!1
"2
#3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

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

 kernel
!bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	"gamma
#beta
$moving_mean
%moving_variance*

&0
'1*

&0
'1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

(0
)1*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*
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
* 
* 
* 
* 
* 
* 
* 

0
1*

I0
J1*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
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
 
0
1
2
3*

0
1*
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
* 
* 
* 
* 

0
1*

W0
X1*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
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
 
0
1
2
3*

0
1*
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
�trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 

$0
%1*

e0
f1*
* 
* 
* 
* 
* 
* 
* 

 0
!1*

 0
!1*
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
* 
* 
* 
 
"0
#1
$2
%3*

"0
#1*
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

0
1*
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
0
1*
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
$0
%1*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedecoder/dense_2/kerneldecoder/dense_2/bias%decoder/decoder_block/conv2d_4/kernel#decoder/decoder_block/conv2d_4/bias1decoder/decoder_block/batch_normalization_4/gamma0decoder/decoder_block/batch_normalization_4/beta7decoder/decoder_block/batch_normalization_4/moving_mean;decoder/decoder_block/batch_normalization_4/moving_variance'decoder/decoder_block_1/conv2d_5/kernel%decoder/decoder_block_1/conv2d_5/bias3decoder/decoder_block_1/batch_normalization_5/gamma2decoder/decoder_block_1/batch_normalization_5/beta9decoder/decoder_block_1/batch_normalization_5/moving_mean=decoder/decoder_block_1/batch_normalization_5/moving_variance'decoder/decoder_block_2/conv2d_6/kernel%decoder/decoder_block_2/conv2d_6/bias3decoder/decoder_block_2/batch_normalization_6/gamma2decoder/decoder_block_2/batch_normalization_6/beta9decoder/decoder_block_2/batch_normalization_6/moving_mean=decoder/decoder_block_2/batch_normalization_6/moving_variancedecoder/conv2d_7/kerneldecoder/conv2d_7/biasdecoder/conv2d_8/kerneldecoder/conv2d_8/biasConst*%
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
 __inference__traced_save_1975458
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder/dense_2/kerneldecoder/dense_2/bias%decoder/decoder_block/conv2d_4/kernel#decoder/decoder_block/conv2d_4/bias1decoder/decoder_block/batch_normalization_4/gamma0decoder/decoder_block/batch_normalization_4/beta7decoder/decoder_block/batch_normalization_4/moving_mean;decoder/decoder_block/batch_normalization_4/moving_variance'decoder/decoder_block_1/conv2d_5/kernel%decoder/decoder_block_1/conv2d_5/bias3decoder/decoder_block_1/batch_normalization_5/gamma2decoder/decoder_block_1/batch_normalization_5/beta9decoder/decoder_block_1/batch_normalization_5/moving_mean=decoder/decoder_block_1/batch_normalization_5/moving_variance'decoder/decoder_block_2/conv2d_6/kernel%decoder/decoder_block_2/conv2d_6/bias3decoder/decoder_block_2/batch_normalization_6/gamma2decoder/decoder_block_2/batch_normalization_6/beta9decoder/decoder_block_2/batch_normalization_6/moving_mean=decoder/decoder_block_2/batch_normalization_6/moving_variancedecoder/conv2d_7/kerneldecoder/conv2d_7/biasdecoder/conv2d_8/kerneldecoder/conv2d_8/bias*$
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
#__inference__traced_restore_1975540��
�
�
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973878
input_tensorA
'conv2d_6_conv2d_readvariableop_resource:@ 6
(conv2d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: 
identity��5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� |
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( �
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+���������������������������@: : : : : : 2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:o k
A
_output_shapes/
-:+���������������������������@
&
_user_specified_nameinput_tensor
�
�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1973758

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1974136
input_1
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�$
	unknown_7:�@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:$

unknown_21:

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
 *A
_output_shapes/
-:+���������������������������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1974085�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
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
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1973517

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
�
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1975105

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1973351

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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975291

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
�
M
1__inference_up_sampling2d_2_layer_call_fn_1974969

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
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1973517�
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
7__inference_batch_normalization_5_layer_call_fn_1975193

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
GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1973477�
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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1973560

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
�
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975065
input_tensorA
'conv2d_6_conv2d_readvariableop_resource:@ 6
(conv2d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: 
identity��5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� |
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( �
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+���������������������������@: : : : : : 2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:o k
A
_output_shapes/
-:+���������������������������@
&
_user_specified_nameinput_tensor
�
�
*__inference_conv2d_8_layer_call_fn_1975094

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1973758�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�5
�

D__inference_decoder_layer_call_and_return_conditional_losses_1973765
input_1$
dense_2_1973603:���
dense_2_1973605:
��1
decoder_block_1973637:��$
decoder_block_1973639:	�$
decoder_block_1973641:	�$
decoder_block_1973643:	�$
decoder_block_1973645:	�$
decoder_block_1973647:	�2
decoder_block_1_1973677:�@%
decoder_block_1_1973679:@%
decoder_block_1_1973681:@%
decoder_block_1_1973683:@%
decoder_block_1_1973685:@%
decoder_block_1_1973687:@1
decoder_block_2_1973717:@ %
decoder_block_2_1973719: %
decoder_block_2_1973721: %
decoder_block_2_1973723: %
decoder_block_2_1973725: %
decoder_block_2_1973727: *
conv2d_7_1973742: 
conv2d_7_1973744:*
conv2d_8_1973759:
conv2d_8_1973761:
identity�� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�%decoder_block/StatefulPartitionedCall�'decoder_block_1/StatefulPartitionedCall�'decoder_block_2/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2_1973603dense_2_1973605*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1973602f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
up_sampling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1973351�
%decoder_block/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0decoder_block_1973637decoder_block_1973639decoder_block_1973641decoder_block_1973643decoder_block_1973645decoder_block_1973647*
Tin
	2*
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
GPU2 *0J 8� *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973636�
up_sampling2d_1/PartitionedCallPartitionedCall.decoder_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1973434�
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0decoder_block_1_1973677decoder_block_1_1973679decoder_block_1_1973681decoder_block_1_1973683decoder_block_1_1973685decoder_block_1_1973687*
Tin
	2*
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
GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973676�
up_sampling2d_2/PartitionedCallPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1973517�
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0decoder_block_2_1973717decoder_block_2_1973719decoder_block_2_1973721decoder_block_2_1973723decoder_block_2_1973725decoder_block_2_1973727*
Tin
	2*
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
GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973716�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_7_1973742conv2d_7_1973744*
Tin
2*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1973741�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_1973759conv2d_8_1973761*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1973758�
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
D__inference_dense_2_layer_call_and_return_conditional_losses_1973602

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
�

�
/__inference_decoder_block_layer_call_fn_1974813
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
 *B
_output_shapes0
.:,����������������������������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973800�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975149

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
�&
�
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974838
input_tensorC
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2Dinput_tensor&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������}
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975167

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
�
�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1973741

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�	
�
1__inference_decoder_block_2_layer_call_fn_1975015
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
 *A
_output_shapes/
-:+��������������������������� *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973878�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+���������������������������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:o k
A
_output_shapes/
-:+���������������������������@
&
_user_specified_nameinput_tensor
�5
�

D__inference_decoder_layer_call_and_return_conditional_losses_1973969
embedding_input$
dense_2_1973909:���
dense_2_1973911:
��1
decoder_block_1973917:��$
decoder_block_1973919:	�$
decoder_block_1973921:	�$
decoder_block_1973923:	�$
decoder_block_1973925:	�$
decoder_block_1973927:	�2
decoder_block_1_1973931:�@%
decoder_block_1_1973933:@%
decoder_block_1_1973935:@%
decoder_block_1_1973937:@%
decoder_block_1_1973939:@%
decoder_block_1_1973941:@1
decoder_block_2_1973945:@ %
decoder_block_2_1973947: %
decoder_block_2_1973949: %
decoder_block_2_1973951: %
decoder_block_2_1973953: %
decoder_block_2_1973955: *
conv2d_7_1973958: 
conv2d_7_1973960:*
conv2d_8_1973963:
conv2d_8_1973965:
identity�� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�%decoder_block/StatefulPartitionedCall�'decoder_block_1/StatefulPartitionedCall�'decoder_block_2/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_2_1973909dense_2_1973911*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1973602f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
up_sampling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1973351�
%decoder_block/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0decoder_block_1973917decoder_block_1973919decoder_block_1973921decoder_block_1973923decoder_block_1973925decoder_block_1973927*
Tin
	2*
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
GPU2 *0J 8� *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973636�
up_sampling2d_1/PartitionedCallPartitionedCall.decoder_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1973434�
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0decoder_block_1_1973931decoder_block_1_1973933decoder_block_1_1973935decoder_block_1_1973937decoder_block_1_1973939decoder_block_1_1973941*
Tin
	2*
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
GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973676�
up_sampling2d_2/PartitionedCallPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1973517�
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0decoder_block_2_1973945decoder_block_2_1973947decoder_block_2_1973949decoder_block_2_1973951decoder_block_2_1973953decoder_block_2_1973955*
Tin
	2*
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
GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973716�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_7_1973958conv2d_7_1973960*
Tin
2*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1973741�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_1973963conv2d_8_1973965*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1973758�
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
�
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1975085

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_1974020
input_1
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�$
	unknown_7:�@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:$

unknown_21:

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
 *A
_output_shapes/
-:+���������������������������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1973969�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
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
�5
�

D__inference_decoder_layer_call_and_return_conditional_losses_1973903
input_1$
dense_2_1973768:���
dense_2_1973770:
��1
decoder_block_1973801:��$
decoder_block_1973803:	�$
decoder_block_1973805:	�$
decoder_block_1973807:	�$
decoder_block_1973809:	�$
decoder_block_1973811:	�2
decoder_block_1_1973840:�@%
decoder_block_1_1973842:@%
decoder_block_1_1973844:@%
decoder_block_1_1973846:@%
decoder_block_1_1973848:@%
decoder_block_1_1973850:@1
decoder_block_2_1973879:@ %
decoder_block_2_1973881: %
decoder_block_2_1973883: %
decoder_block_2_1973885: %
decoder_block_2_1973887: %
decoder_block_2_1973889: *
conv2d_7_1973892: 
conv2d_7_1973894:*
conv2d_8_1973897:
conv2d_8_1973899:
identity�� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�%decoder_block/StatefulPartitionedCall�'decoder_block_1/StatefulPartitionedCall�'decoder_block_2/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2_1973768dense_2_1973770*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1973602f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
up_sampling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1973351�
%decoder_block/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0decoder_block_1973801decoder_block_1973803decoder_block_1973805decoder_block_1973807decoder_block_1973809decoder_block_1973811*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973800�
up_sampling2d_1/PartitionedCallPartitionedCall.decoder_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1973434�
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0decoder_block_1_1973840decoder_block_1_1973842decoder_block_1_1973844decoder_block_1_1973846decoder_block_1_1973848decoder_block_1_1973850*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973839�
up_sampling2d_2/PartitionedCallPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1973517�
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0decoder_block_2_1973879decoder_block_2_1973881decoder_block_2_1973883decoder_block_2_1973885decoder_block_2_1973887decoder_block_2_1973889*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973878�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_7_1973892conv2d_7_1973894*
Tin
2*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1973741�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_1973897conv2d_8_1973899*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1973758�
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974863
input_tensorC
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2Dinput_tensor&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������}
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1974779

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
�
K
/__inference_up_sampling2d_layer_call_fn_1974767

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
GPU2 *0J 8� *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1973351�
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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975273

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
)__inference_decoder_layer_call_fn_1974538
embedding_input
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�$
	unknown_7:�@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:$

unknown_21:

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
 *A
_output_shapes/
-:+���������������������������*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1974085�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
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
�&
�
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974939
input_tensorB
'conv2d_5_conv2d_readvariableop_resource:�@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity��$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_5/Conv2DConv2Dinput_tensor&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�&
�
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973636
input_tensorC
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_4/AssignNewValue�&batch_normalization_4/AssignNewValue_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2Dinput_tensor&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������}
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�	
�
7__inference_batch_normalization_6_layer_call_fn_1975242

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
GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1973542�
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
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1973394

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
��
�
 __inference__traced_save_1975458
file_prefixB
-read_disablecopyonread_decoder_dense_2_kernel:���=
-read_1_disablecopyonread_decoder_dense_2_bias:
��Z
>read_2_disablecopyonread_decoder_decoder_block_conv2d_4_kernel:��K
<read_3_disablecopyonread_decoder_decoder_block_conv2d_4_bias:	�Y
Jread_4_disablecopyonread_decoder_decoder_block_batch_normalization_4_gamma:	�X
Iread_5_disablecopyonread_decoder_decoder_block_batch_normalization_4_beta:	�_
Pread_6_disablecopyonread_decoder_decoder_block_batch_normalization_4_moving_mean:	�c
Tread_7_disablecopyonread_decoder_decoder_block_batch_normalization_4_moving_variance:	�[
@read_8_disablecopyonread_decoder_decoder_block_1_conv2d_5_kernel:�@L
>read_9_disablecopyonread_decoder_decoder_block_1_conv2d_5_bias:@[
Mread_10_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_gamma:@Z
Lread_11_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_beta:@a
Sread_12_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_moving_mean:@e
Wread_13_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_moving_variance:@[
Aread_14_disablecopyonread_decoder_decoder_block_2_conv2d_6_kernel:@ M
?read_15_disablecopyonread_decoder_decoder_block_2_conv2d_6_bias: [
Mread_16_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_gamma: Z
Lread_17_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_beta: a
Sread_18_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_moving_mean: e
Wread_19_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_moving_variance: K
1read_20_disablecopyonread_decoder_conv2d_7_kernel: =
/read_21_disablecopyonread_decoder_conv2d_7_bias:K
1read_22_disablecopyonread_decoder_conv2d_8_kernel:=
/read_23_disablecopyonread_decoder_conv2d_8_bias:
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
Read/DisableCopyOnReadDisableCopyOnRead-read_disablecopyonread_decoder_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp-read_disablecopyonread_decoder_dense_2_kernel^Read/DisableCopyOnRead"/device:CPU:0*!
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
Read_1/DisableCopyOnReadDisableCopyOnRead-read_1_disablecopyonread_decoder_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp-read_1_disablecopyonread_decoder_dense_2_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
Read_2/DisableCopyOnReadDisableCopyOnRead>read_2_disablecopyonread_decoder_decoder_block_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp>read_2_disablecopyonread_decoder_decoder_block_conv2d_4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0w

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��m

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_3/DisableCopyOnReadDisableCopyOnRead<read_3_disablecopyonread_decoder_decoder_block_conv2d_4_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp<read_3_disablecopyonread_decoder_decoder_block_conv2d_4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnReadJread_4_disablecopyonread_decoder_decoder_block_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpJread_4_disablecopyonread_decoder_decoder_block_batch_normalization_4_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_5/DisableCopyOnReadDisableCopyOnReadIread_5_disablecopyonread_decoder_decoder_block_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpIread_5_disablecopyonread_decoder_decoder_block_batch_normalization_4_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_6/DisableCopyOnReadDisableCopyOnReadPread_6_disablecopyonread_decoder_decoder_block_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOpPread_6_disablecopyonread_decoder_decoder_block_batch_normalization_4_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_7/DisableCopyOnReadDisableCopyOnReadTread_7_disablecopyonread_decoder_decoder_block_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOpTread_7_disablecopyonread_decoder_decoder_block_batch_normalization_4_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnRead@read_8_disablecopyonread_decoder_decoder_block_1_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp@read_8_disablecopyonread_decoder_decoder_block_1_conv2d_5_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�@*
dtype0w
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�@n
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*'
_output_shapes
:�@�
Read_9/DisableCopyOnReadDisableCopyOnRead>read_9_disablecopyonread_decoder_decoder_block_1_conv2d_5_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp>read_9_disablecopyonread_decoder_decoder_block_1_conv2d_5_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_10/DisableCopyOnReadDisableCopyOnReadMread_10_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpMread_10_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnReadLread_11_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpLread_11_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_12/DisableCopyOnReadDisableCopyOnReadSread_12_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpSread_12_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_13/DisableCopyOnReadDisableCopyOnReadWread_13_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpWread_13_disablecopyonread_decoder_decoder_block_1_batch_normalization_5_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_14/DisableCopyOnReadDisableCopyOnReadAread_14_disablecopyonread_decoder_decoder_block_2_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpAread_14_disablecopyonread_decoder_decoder_block_2_conv2d_6_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ �
Read_15/DisableCopyOnReadDisableCopyOnRead?read_15_disablecopyonread_decoder_decoder_block_2_conv2d_6_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp?read_15_disablecopyonread_decoder_decoder_block_2_conv2d_6_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnReadMread_16_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpMread_16_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnReadLread_17_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpLread_17_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnReadSread_18_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOpSread_18_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnReadWread_19_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOpWread_19_disablecopyonread_decoder_decoder_block_2_batch_normalization_6_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_decoder_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_decoder_conv2d_7_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_decoder_conv2d_7_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_decoder_conv2d_7_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_22/DisableCopyOnReadDisableCopyOnRead1read_22_disablecopyonread_decoder_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp1read_22_disablecopyonread_decoder_conv2d_8_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:*
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_decoder_conv2d_8_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_decoder_conv2d_8_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
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
�&
�
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975040
input_tensorA
'conv2d_6_conv2d_readvariableop_resource:@ 6
(conv2d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: 
identity��$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� |
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+���������������������������@: : : : : : 2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:o k
A
_output_shapes/
-:+���������������������������@
&
_user_specified_nameinput_tensor
�
�
)__inference_dense_2_layer_call_fn_1974751

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
D__inference_dense_2_layer_call_and_return_conditional_losses_1973602q
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
�
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1974981

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

�
D__inference_dense_2_layer_call_and_return_conditional_losses_1974762

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
�	
�
1__inference_decoder_block_1_layer_call_fn_1974897
input_tensor"
unknown:�@
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
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973676�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�	
�
7__inference_batch_normalization_6_layer_call_fn_1975255

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
GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1973560�
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
�
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1974880

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
�5
�

D__inference_decoder_layer_call_and_return_conditional_losses_1974085
embedding_input$
dense_2_1974025:���
dense_2_1974027:
��1
decoder_block_1974033:��$
decoder_block_1974035:	�$
decoder_block_1974037:	�$
decoder_block_1974039:	�$
decoder_block_1974041:	�$
decoder_block_1974043:	�2
decoder_block_1_1974047:�@%
decoder_block_1_1974049:@%
decoder_block_1_1974051:@%
decoder_block_1_1974053:@%
decoder_block_1_1974055:@%
decoder_block_1_1974057:@1
decoder_block_2_1974061:@ %
decoder_block_2_1974063: %
decoder_block_2_1974065: %
decoder_block_2_1974067: %
decoder_block_2_1974069: %
decoder_block_2_1974071: *
conv2d_7_1974074: 
conv2d_7_1974076:*
conv2d_8_1974079:
conv2d_8_1974081:
identity�� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�%decoder_block/StatefulPartitionedCall�'decoder_block_1/StatefulPartitionedCall�'decoder_block_2/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_2_1974025dense_2_1974027*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1973602f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:�����������
up_sampling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1973351�
%decoder_block/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0decoder_block_1974033decoder_block_1974035decoder_block_1974037decoder_block_1974039decoder_block_1974041decoder_block_1974043*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973800�
up_sampling2d_1/PartitionedCallPartitionedCall.decoder_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1973434�
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0decoder_block_1_1974047decoder_block_1_1974049decoder_block_1_1974051decoder_block_1_1974053decoder_block_1_1974055decoder_block_1_1974057*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973839�
up_sampling2d_2/PartitionedCallPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1973517�
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0decoder_block_2_1974061decoder_block_2_1974063decoder_block_2_1974065decoder_block_2_1974067decoder_block_2_1974069decoder_block_2_1974071*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973878�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_7_1974074conv2d_7_1974076*
Tin
2*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1973741�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_1974079conv2d_8_1974081*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1973758�
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�&
�
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973676
input_tensorB
'conv2d_5_conv2d_readvariableop_resource:�@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity��$batch_normalization_5/AssignNewValue�&batch_normalization_5/AssignNewValue_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_5/Conv2DConv2Dinput_tensor&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�	
�
1__inference_decoder_block_2_layer_call_fn_1974998
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
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973716�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+���������������������������@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:o k
A
_output_shapes/
-:+���������������������������@
&
_user_specified_nameinput_tensor
�
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1973477

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
�
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1973434

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
�
M
1__inference_up_sampling2d_1_layer_call_fn_1974868

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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1973434�
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
�
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975229

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
��
�
"__inference__wrapped_model_1973338
input_1C
.decoder_dense_2_matmul_readvariableop_resource:���?
/decoder_dense_2_biasadd_readvariableop_resource:
��Y
=decoder_decoder_block_conv2d_4_conv2d_readvariableop_resource:��M
>decoder_decoder_block_conv2d_4_biasadd_readvariableop_resource:	�R
Cdecoder_decoder_block_batch_normalization_4_readvariableop_resource:	�T
Edecoder_decoder_block_batch_normalization_4_readvariableop_1_resource:	�c
Tdecoder_decoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�e
Vdecoder_decoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�Z
?decoder_decoder_block_1_conv2d_5_conv2d_readvariableop_resource:�@N
@decoder_decoder_block_1_conv2d_5_biasadd_readvariableop_resource:@S
Edecoder_decoder_block_1_batch_normalization_5_readvariableop_resource:@U
Gdecoder_decoder_block_1_batch_normalization_5_readvariableop_1_resource:@d
Vdecoder_decoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@f
Xdecoder_decoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@Y
?decoder_decoder_block_2_conv2d_6_conv2d_readvariableop_resource:@ N
@decoder_decoder_block_2_conv2d_6_biasadd_readvariableop_resource: S
Edecoder_decoder_block_2_batch_normalization_6_readvariableop_resource: U
Gdecoder_decoder_block_2_batch_normalization_6_readvariableop_1_resource: d
Vdecoder_decoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: f
Xdecoder_decoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: I
/decoder_conv2d_7_conv2d_readvariableop_resource: >
0decoder_conv2d_7_biasadd_readvariableop_resource:I
/decoder_conv2d_8_conv2d_readvariableop_resource:>
0decoder_conv2d_8_biasadd_readvariableop_resource:
identity��'decoder/conv2d_7/BiasAdd/ReadVariableOp�&decoder/conv2d_7/Conv2D/ReadVariableOp�'decoder/conv2d_8/BiasAdd/ReadVariableOp�&decoder/conv2d_8/Conv2D/ReadVariableOp�Kdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�Mdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�:decoder/decoder_block/batch_normalization_4/ReadVariableOp�<decoder/decoder_block/batch_normalization_4/ReadVariableOp_1�5decoder/decoder_block/conv2d_4/BiasAdd/ReadVariableOp�4decoder/decoder_block/conv2d_4/Conv2D/ReadVariableOp�Mdecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�Odecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�<decoder/decoder_block_1/batch_normalization_5/ReadVariableOp�>decoder/decoder_block_1/batch_normalization_5/ReadVariableOp_1�7decoder/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp�6decoder/decoder_block_1/conv2d_5/Conv2D/ReadVariableOp�Mdecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Odecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�<decoder/decoder_block_2/batch_normalization_6/ReadVariableOp�>decoder/decoder_block_2/batch_normalization_6/ReadVariableOp_1�7decoder/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp�6decoder/decoder_block_2/conv2d_6/Conv2D/ReadVariableOp�&decoder/dense_2/BiasAdd/ReadVariableOp�%decoder/dense_2/MatMul/ReadVariableOp�
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
decoder/dense_2/MatMulMatMulinput_1-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������r
decoder/dense_2/ReluRelu decoder/dense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������n
decoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
decoder/ReshapeReshape"decoder/dense_2/Relu:activations:0decoder/Reshape/shape:output:0*
T0*0
_output_shapes
:����������l
decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      n
decoder/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
decoder/up_sampling2d/mulMul$decoder/up_sampling2d/Const:output:0&decoder/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:�
2decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbordecoder/Reshape:output:0decoder/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
4decoder/decoder_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp=decoder_decoder_block_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%decoder/decoder_block/conv2d_4/Conv2DConv2DCdecoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0<decoder/decoder_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
5decoder/decoder_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp>decoder_decoder_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&decoder/decoder_block/conv2d_4/BiasAddBiasAdd.decoder/decoder_block/conv2d_4/Conv2D:output:0=decoder/decoder_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
#decoder/decoder_block/conv2d_4/ReluRelu/decoder/decoder_block/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
:decoder/decoder_block/batch_normalization_4/ReadVariableOpReadVariableOpCdecoder_decoder_block_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<decoder/decoder_block/batch_normalization_4/ReadVariableOp_1ReadVariableOpEdecoder_decoder_block_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Kdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpTdecoder_decoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Mdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVdecoder_decoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
<decoder/decoder_block/batch_normalization_4/FusedBatchNormV3FusedBatchNormV31decoder/decoder_block/conv2d_4/Relu:activations:0Bdecoder/decoder_block/batch_normalization_4/ReadVariableOp:value:0Ddecoder/decoder_block/batch_normalization_4/ReadVariableOp_1:value:0Sdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Udecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( n
decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      p
decoder/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
decoder/up_sampling2d_1/mulMul&decoder/up_sampling2d_1/Const:output:0(decoder/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:�
4decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor@decoder/decoder_block/batch_normalization_4/FusedBatchNormV3:y:0decoder/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
6decoder/decoder_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp?decoder_decoder_block_1_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
'decoder/decoder_block_1/conv2d_5/Conv2DConv2DEdecoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0>decoder/decoder_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
7decoder/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp@decoder_decoder_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
(decoder/decoder_block_1/conv2d_5/BiasAddBiasAdd0decoder/decoder_block_1/conv2d_5/Conv2D:output:0?decoder/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
%decoder/decoder_block_1/conv2d_5/ReluRelu1decoder/decoder_block_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
<decoder/decoder_block_1/batch_normalization_5/ReadVariableOpReadVariableOpEdecoder_decoder_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
>decoder/decoder_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOpGdecoder_decoder_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Mdecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpVdecoder_decoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Odecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXdecoder_decoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
>decoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV33decoder/decoder_block_1/conv2d_5/Relu:activations:0Ddecoder/decoder_block_1/batch_normalization_5/ReadVariableOp:value:0Fdecoder/decoder_block_1/batch_normalization_5/ReadVariableOp_1:value:0Udecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Wdecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( n
decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        p
decoder/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
decoder/up_sampling2d_2/mulMul&decoder/up_sampling2d_2/Const:output:0(decoder/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:�
4decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborBdecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3:y:0decoder/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:�����������@*
half_pixel_centers(�
6decoder/decoder_block_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp?decoder_decoder_block_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
'decoder/decoder_block_2/conv2d_6/Conv2DConv2DEdecoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0>decoder/decoder_block_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
7decoder/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp@decoder_decoder_block_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
(decoder/decoder_block_2/conv2d_6/BiasAddBiasAdd0decoder/decoder_block_2/conv2d_6/Conv2D:output:0?decoder/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
%decoder/decoder_block_2/conv2d_6/ReluRelu1decoder/decoder_block_2/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
<decoder/decoder_block_2/batch_normalization_6/ReadVariableOpReadVariableOpEdecoder_decoder_block_2_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
>decoder/decoder_block_2/batch_normalization_6/ReadVariableOp_1ReadVariableOpGdecoder_decoder_block_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Mdecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpVdecoder_decoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Odecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXdecoder_decoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
>decoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV33decoder/decoder_block_2/conv2d_6/Relu:activations:0Ddecoder/decoder_block_2/batch_normalization_6/ReadVariableOp:value:0Fdecoder/decoder_block_2/batch_normalization_6/ReadVariableOp_1:value:0Udecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Wdecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
is_training( �
&decoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
decoder/conv2d_7/Conv2DConv2DBdecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3:y:0.decoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
'decoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
decoder/conv2d_7/BiasAddBiasAdd decoder/conv2d_7/Conv2D:output:0/decoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������|
decoder/conv2d_7/ReluRelu!decoder/conv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:������������
&decoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
decoder/conv2d_8/Conv2DConv2D#decoder/conv2d_7/Relu:activations:0.decoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
'decoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
decoder/conv2d_8/BiasAddBiasAdd decoder/conv2d_8/Conv2D:output:0/decoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:������������
decoder/conv2d_8/SigmoidSigmoid!decoder/conv2d_8/BiasAdd:output:0*
T0*1
_output_shapes
:�����������u
IdentityIdentitydecoder/conv2d_8/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp(^decoder/conv2d_7/BiasAdd/ReadVariableOp'^decoder/conv2d_7/Conv2D/ReadVariableOp(^decoder/conv2d_8/BiasAdd/ReadVariableOp'^decoder/conv2d_8/Conv2D/ReadVariableOpL^decoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpN^decoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1;^decoder/decoder_block/batch_normalization_4/ReadVariableOp=^decoder/decoder_block/batch_normalization_4/ReadVariableOp_16^decoder/decoder_block/conv2d_4/BiasAdd/ReadVariableOp5^decoder/decoder_block/conv2d_4/Conv2D/ReadVariableOpN^decoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpP^decoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1=^decoder/decoder_block_1/batch_normalization_5/ReadVariableOp?^decoder/decoder_block_1/batch_normalization_5/ReadVariableOp_18^decoder/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp7^decoder/decoder_block_1/conv2d_5/Conv2D/ReadVariableOpN^decoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpP^decoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=^decoder/decoder_block_2/batch_normalization_6/ReadVariableOp?^decoder/decoder_block_2/batch_normalization_6/ReadVariableOp_18^decoder/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp7^decoder/decoder_block_2/conv2d_6/Conv2D/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2R
'decoder/conv2d_7/BiasAdd/ReadVariableOp'decoder/conv2d_7/BiasAdd/ReadVariableOp2P
&decoder/conv2d_7/Conv2D/ReadVariableOp&decoder/conv2d_7/Conv2D/ReadVariableOp2R
'decoder/conv2d_8/BiasAdd/ReadVariableOp'decoder/conv2d_8/BiasAdd/ReadVariableOp2P
&decoder/conv2d_8/Conv2D/ReadVariableOp&decoder/conv2d_8/Conv2D/ReadVariableOp2�
Mdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Mdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12�
Kdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpKdecoder/decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2|
<decoder/decoder_block/batch_normalization_4/ReadVariableOp_1<decoder/decoder_block/batch_normalization_4/ReadVariableOp_12x
:decoder/decoder_block/batch_normalization_4/ReadVariableOp:decoder/decoder_block/batch_normalization_4/ReadVariableOp2n
5decoder/decoder_block/conv2d_4/BiasAdd/ReadVariableOp5decoder/decoder_block/conv2d_4/BiasAdd/ReadVariableOp2l
4decoder/decoder_block/conv2d_4/Conv2D/ReadVariableOp4decoder/decoder_block/conv2d_4/Conv2D/ReadVariableOp2�
Odecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Odecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12�
Mdecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpMdecoder/decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
>decoder/decoder_block_1/batch_normalization_5/ReadVariableOp_1>decoder/decoder_block_1/batch_normalization_5/ReadVariableOp_12|
<decoder/decoder_block_1/batch_normalization_5/ReadVariableOp<decoder/decoder_block_1/batch_normalization_5/ReadVariableOp2r
7decoder/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp7decoder/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp2p
6decoder/decoder_block_1/conv2d_5/Conv2D/ReadVariableOp6decoder/decoder_block_1/conv2d_5/Conv2D/ReadVariableOp2�
Odecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Odecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12�
Mdecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpMdecoder/decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
>decoder/decoder_block_2/batch_normalization_6/ReadVariableOp_1>decoder/decoder_block_2/batch_normalization_6/ReadVariableOp_12|
<decoder/decoder_block_2/batch_normalization_6/ReadVariableOp<decoder/decoder_block_2/batch_normalization_6/ReadVariableOp2r
7decoder/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp7decoder/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp2p
6decoder/decoder_block_2/conv2d_6/Conv2D/ReadVariableOp6decoder/decoder_block_2/conv2d_6/Conv2D/ReadVariableOp2P
&decoder/dense_2/BiasAdd/ReadVariableOp&decoder/dense_2/BiasAdd/ReadVariableOp2N
%decoder/dense_2/MatMul/ReadVariableOp%decoder/dense_2/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974964
input_tensorB
'conv2d_5_conv2d_readvariableop_resource:�@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity��5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_5/Conv2DConv2Dinput_tensor&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( �
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�n
�
#__inference__traced_restore_1975540
file_prefix<
'assignvariableop_decoder_dense_2_kernel:���7
'assignvariableop_1_decoder_dense_2_bias:
��T
8assignvariableop_2_decoder_decoder_block_conv2d_4_kernel:��E
6assignvariableop_3_decoder_decoder_block_conv2d_4_bias:	�S
Dassignvariableop_4_decoder_decoder_block_batch_normalization_4_gamma:	�R
Cassignvariableop_5_decoder_decoder_block_batch_normalization_4_beta:	�Y
Jassignvariableop_6_decoder_decoder_block_batch_normalization_4_moving_mean:	�]
Nassignvariableop_7_decoder_decoder_block_batch_normalization_4_moving_variance:	�U
:assignvariableop_8_decoder_decoder_block_1_conv2d_5_kernel:�@F
8assignvariableop_9_decoder_decoder_block_1_conv2d_5_bias:@U
Gassignvariableop_10_decoder_decoder_block_1_batch_normalization_5_gamma:@T
Fassignvariableop_11_decoder_decoder_block_1_batch_normalization_5_beta:@[
Massignvariableop_12_decoder_decoder_block_1_batch_normalization_5_moving_mean:@_
Qassignvariableop_13_decoder_decoder_block_1_batch_normalization_5_moving_variance:@U
;assignvariableop_14_decoder_decoder_block_2_conv2d_6_kernel:@ G
9assignvariableop_15_decoder_decoder_block_2_conv2d_6_bias: U
Gassignvariableop_16_decoder_decoder_block_2_batch_normalization_6_gamma: T
Fassignvariableop_17_decoder_decoder_block_2_batch_normalization_6_beta: [
Massignvariableop_18_decoder_decoder_block_2_batch_normalization_6_moving_mean: _
Qassignvariableop_19_decoder_decoder_block_2_batch_normalization_6_moving_variance: E
+assignvariableop_20_decoder_conv2d_7_kernel: 7
)assignvariableop_21_decoder_conv2d_7_bias:E
+assignvariableop_22_decoder_conv2d_8_kernel:7
)assignvariableop_23_decoder_conv2d_8_bias:
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
AssignVariableOpAssignVariableOp'assignvariableop_decoder_dense_2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp'assignvariableop_1_decoder_dense_2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp8assignvariableop_2_decoder_decoder_block_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_decoder_decoder_block_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpDassignvariableop_4_decoder_decoder_block_batch_normalization_4_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpCassignvariableop_5_decoder_decoder_block_batch_normalization_4_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpJassignvariableop_6_decoder_decoder_block_batch_normalization_4_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpNassignvariableop_7_decoder_decoder_block_batch_normalization_4_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp:assignvariableop_8_decoder_decoder_block_1_conv2d_5_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp8assignvariableop_9_decoder_decoder_block_1_conv2d_5_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpGassignvariableop_10_decoder_decoder_block_1_batch_normalization_5_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpFassignvariableop_11_decoder_decoder_block_1_batch_normalization_5_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpMassignvariableop_12_decoder_decoder_block_1_batch_normalization_5_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpQassignvariableop_13_decoder_decoder_block_1_batch_normalization_5_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp;assignvariableop_14_decoder_decoder_block_2_conv2d_6_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp9assignvariableop_15_decoder_decoder_block_2_conv2d_6_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpGassignvariableop_16_decoder_decoder_block_2_batch_normalization_6_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpFassignvariableop_17_decoder_decoder_block_2_batch_normalization_6_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpMassignvariableop_18_decoder_decoder_block_2_batch_normalization_6_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpQassignvariableop_19_decoder_decoder_block_2_batch_normalization_6_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_decoder_conv2d_7_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_decoder_conv2d_7_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp+assignvariableop_22_decoder_conv2d_8_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_decoder_conv2d_8_biasIdentity_23:output:0"/device:CPU:0*&
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
�
�
)__inference_decoder_layer_call_fn_1974485
embedding_input
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�$
	unknown_7:�@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:$

unknown_21:

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
 *A
_output_shapes/
-:+���������������������������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1973969�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
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
�
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973839
input_tensorB
'conv2d_5_conv2d_readvariableop_resource:�@6
(conv2d_5_biasadd_readvariableop_resource:@;
-batch_normalization_5_readvariableop_resource:@=
/batch_normalization_5_readvariableop_1_resource:@L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@
identity��5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�conv2d_5/BiasAdd/ReadVariableOp�conv2d_5/Conv2D/ReadVariableOp�
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
conv2d_5/Conv2DConv2Dinput_tensor&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( �
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�	
�
7__inference_batch_normalization_4_layer_call_fn_1975131

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
GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1973394�
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
�
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975211

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
�&
�
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1973716
input_tensorA
'conv2d_6_conv2d_readvariableop_resource:@ 6
(conv2d_6_biasadd_readvariableop_resource: ;
-batch_normalization_6_readvariableop_resource: =
/batch_normalization_6_readvariableop_1_resource: L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: 
identity��$batch_normalization_6/AssignNewValue�&batch_normalization_6/AssignNewValue_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� |
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� �
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:+���������������������������@: : : : : : 2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:o k
A
_output_shapes/
-:+���������������������������@
&
_user_specified_nameinput_tensor
�
�
*__inference_conv2d_7_layer_call_fn_1975074

inputs!
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2 *0J 8� *N
fIRG
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1973741�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�	
�
7__inference_batch_normalization_4_layer_call_fn_1975118

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
GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1973376�
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
�
�
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1973376

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
�
�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1973542

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
7__inference_batch_normalization_5_layer_call_fn_1975180

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
GPU2 *0J 8� *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1973459�
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
�	
�
1__inference_decoder_block_1_layer_call_fn_1974914
input_tensor"
unknown:�@
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
 *A
_output_shapes/
-:+���������������������������@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1973839�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�	
�
/__inference_decoder_block_layer_call_fn_1974796
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
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973636�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
�
�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1973459

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
��
�
D__inference_decoder_layer_call_and_return_conditional_losses_1974742
embedding_input;
&dense_2_matmul_readvariableop_resource:���7
'dense_2_biasadd_readvariableop_resource:
��Q
5decoder_block_conv2d_4_conv2d_readvariableop_resource:��E
6decoder_block_conv2d_4_biasadd_readvariableop_resource:	�J
;decoder_block_batch_normalization_4_readvariableop_resource:	�L
=decoder_block_batch_normalization_4_readvariableop_1_resource:	�[
Ldecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�]
Ndecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�R
7decoder_block_1_conv2d_5_conv2d_readvariableop_resource:�@F
8decoder_block_1_conv2d_5_biasadd_readvariableop_resource:@K
=decoder_block_1_batch_normalization_5_readvariableop_resource:@M
?decoder_block_1_batch_normalization_5_readvariableop_1_resource:@\
Ndecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@^
Pdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@Q
7decoder_block_2_conv2d_6_conv2d_readvariableop_resource:@ F
8decoder_block_2_conv2d_6_biasadd_readvariableop_resource: K
=decoder_block_2_batch_normalization_6_readvariableop_resource: M
?decoder_block_2_batch_normalization_6_readvariableop_1_resource: \
Ndecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: ^
Pdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource:A
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�Cdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�2decoder_block/batch_normalization_4/ReadVariableOp�4decoder_block/batch_normalization_4/ReadVariableOp_1�-decoder_block/conv2d_4/BiasAdd/ReadVariableOp�,decoder_block/conv2d_4/Conv2D/ReadVariableOp�Edecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�4decoder_block_1/batch_normalization_5/ReadVariableOp�6decoder_block_1/batch_normalization_5/ReadVariableOp_1�/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp�.decoder_block_1/conv2d_5/Conv2D/ReadVariableOp�Edecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�4decoder_block_2/batch_normalization_6/ReadVariableOp�6decoder_block_2/batch_normalization_6/ReadVariableOp_1�/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp�.decoder_block_2/conv2d_6/Conv2D/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
dense_2/MatMulMatMulembedding_input%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������b
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshapedense_2/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:����������d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:�
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborReshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
,decoder_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp5decoder_block_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
decoder_block/conv2d_4/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:04decoder_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
-decoder_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6decoder_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder_block/conv2d_4/BiasAddBiasAdd&decoder_block/conv2d_4/Conv2D:output:05decoder_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
decoder_block/conv2d_4/ReluRelu'decoder_block/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
2decoder_block/batch_normalization_4/ReadVariableOpReadVariableOp;decoder_block_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4decoder_block/batch_normalization_4/ReadVariableOp_1ReadVariableOp=decoder_block_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Cdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpLdecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNdecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
4decoder_block/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3)decoder_block/conv2d_4/Relu:activations:0:decoder_block/batch_normalization_4/ReadVariableOp:value:0<decoder_block/batch_normalization_4/ReadVariableOp_1:value:0Kdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Mdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor8decoder_block/batch_normalization_4/FusedBatchNormV3:y:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
.decoder_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7decoder_block_1_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
decoder_block_1/conv2d_5/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:06decoder_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 decoder_block_1/conv2d_5/BiasAddBiasAdd(decoder_block_1/conv2d_5/Conv2D:output:07decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
decoder_block_1/conv2d_5/ReluRelu)decoder_block_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
4decoder_block_1/batch_normalization_5/ReadVariableOpReadVariableOp=decoder_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
6decoder_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp?decoder_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Edecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6decoder_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+decoder_block_1/conv2d_5/Relu:activations:0<decoder_block_1/batch_normalization_5/ReadVariableOp:value:0>decoder_block_1/batch_normalization_5/ReadVariableOp_1:value:0Mdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
is_training( f
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor:decoder_block_1/batch_normalization_5/FusedBatchNormV3:y:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:�����������@*
half_pixel_centers(�
.decoder_block_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp7decoder_block_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
decoder_block_2/conv2d_6/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:06decoder_block_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 decoder_block_2/conv2d_6/BiasAddBiasAdd(decoder_block_2/conv2d_6/Conv2D:output:07decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
decoder_block_2/conv2d_6/ReluRelu)decoder_block_2/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
4decoder_block_2/batch_normalization_6/ReadVariableOpReadVariableOp=decoder_block_2_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
6decoder_block_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp?decoder_block_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Edecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6decoder_block_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3+decoder_block_2/conv2d_6/Relu:activations:0<decoder_block_2/batch_normalization_6/ReadVariableOp:value:0>decoder_block_2/batch_normalization_6/ReadVariableOp_1:value:0Mdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
is_training( �
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_7/Conv2DConv2D:decoder_block_2/batch_normalization_6/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������l
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:������������
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������r
conv2d_8/SigmoidSigmoidconv2d_8/BiasAdd:output:0*
T0*1
_output_shapes
:�����������m
IdentityIdentityconv2d_8/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������

NoOpNoOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOpD^decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpF^decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_13^decoder_block/batch_normalization_4/ReadVariableOp5^decoder_block/batch_normalization_4/ReadVariableOp_1.^decoder_block/conv2d_4/BiasAdd/ReadVariableOp-^decoder_block/conv2d_4/Conv2D/ReadVariableOpF^decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpH^decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_15^decoder_block_1/batch_normalization_5/ReadVariableOp7^decoder_block_1/batch_normalization_5/ReadVariableOp_10^decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp/^decoder_block_1/conv2d_5/Conv2D/ReadVariableOpF^decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_15^decoder_block_2/batch_normalization_6/ReadVariableOp7^decoder_block_2/batch_normalization_6/ReadVariableOp_10^decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp/^decoder_block_2/conv2d_6/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2�
Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12�
Cdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpCdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2l
4decoder_block/batch_normalization_4/ReadVariableOp_14decoder_block/batch_normalization_4/ReadVariableOp_12h
2decoder_block/batch_normalization_4/ReadVariableOp2decoder_block/batch_normalization_4/ReadVariableOp2^
-decoder_block/conv2d_4/BiasAdd/ReadVariableOp-decoder_block/conv2d_4/BiasAdd/ReadVariableOp2\
,decoder_block/conv2d_4/Conv2D/ReadVariableOp,decoder_block/conv2d_4/Conv2D/ReadVariableOp2�
Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12�
Edecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpEdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_1/batch_normalization_5/ReadVariableOp_16decoder_block_1/batch_normalization_5/ReadVariableOp_12l
4decoder_block_1/batch_normalization_5/ReadVariableOp4decoder_block_1/batch_normalization_5/ReadVariableOp2b
/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp2`
.decoder_block_1/conv2d_5/Conv2D/ReadVariableOp.decoder_block_1/conv2d_5/Conv2D/ReadVariableOp2�
Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12�
Edecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_2/batch_normalization_6/ReadVariableOp_16decoder_block_2/batch_normalization_6/ReadVariableOp_12l
4decoder_block_2/batch_normalization_6/ReadVariableOp4decoder_block_2/batch_normalization_6/ReadVariableOp2b
/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp2`
.decoder_block_2/conv2d_6/Conv2D/ReadVariableOp.decoder_block_2/conv2d_6/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input
�
�
%__inference_signature_wrapper_1974432
input_1
unknown:���
	unknown_0:
��%
	unknown_1:��
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�$
	unknown_7:�@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@$

unknown_13:@ 

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: $

unknown_19: 

unknown_20:$

unknown_21:

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
"__inference__wrapped_model_1973338y
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
�
J__inference_decoder_block_layer_call_and_return_conditional_losses_1973800
input_tensorC
'conv2d_4_conv2d_readvariableop_resource:��7
(conv2d_4_biasadd_readvariableop_resource:	�<
-batch_normalization_4_readvariableop_resource:	�>
/batch_normalization_4_readvariableop_1_resource:	�M
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�conv2d_4/BiasAdd/ReadVariableOp�conv2d_4/Conv2D/ReadVariableOp�
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_4/Conv2DConv2Dinput_tensor&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������}
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*B
_output_shapes0
.:,�����������������������������
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( �
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,����������������������������: : : : : : 2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,����������������������������
&
_user_specified_nameinput_tensor
ܪ
�
D__inference_decoder_layer_call_and_return_conditional_losses_1974640
embedding_input;
&dense_2_matmul_readvariableop_resource:���7
'dense_2_biasadd_readvariableop_resource:
��Q
5decoder_block_conv2d_4_conv2d_readvariableop_resource:��E
6decoder_block_conv2d_4_biasadd_readvariableop_resource:	�J
;decoder_block_batch_normalization_4_readvariableop_resource:	�L
=decoder_block_batch_normalization_4_readvariableop_1_resource:	�[
Ldecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�]
Ndecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�R
7decoder_block_1_conv2d_5_conv2d_readvariableop_resource:�@F
8decoder_block_1_conv2d_5_biasadd_readvariableop_resource:@K
=decoder_block_1_batch_normalization_5_readvariableop_resource:@M
?decoder_block_1_batch_normalization_5_readvariableop_1_resource:@\
Ndecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@^
Pdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@Q
7decoder_block_2_conv2d_6_conv2d_readvariableop_resource:@ F
8decoder_block_2_conv2d_6_biasadd_readvariableop_resource: K
=decoder_block_2_batch_normalization_6_readvariableop_resource: M
?decoder_block_2_batch_normalization_6_readvariableop_1_resource: \
Ndecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource: ^
Pdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_7_conv2d_readvariableop_resource: 6
(conv2d_7_biasadd_readvariableop_resource:A
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:
identity��conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�2decoder_block/batch_normalization_4/AssignNewValue�4decoder_block/batch_normalization_4/AssignNewValue_1�Cdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�2decoder_block/batch_normalization_4/ReadVariableOp�4decoder_block/batch_normalization_4/ReadVariableOp_1�-decoder_block/conv2d_4/BiasAdd/ReadVariableOp�,decoder_block/conv2d_4/Conv2D/ReadVariableOp�4decoder_block_1/batch_normalization_5/AssignNewValue�6decoder_block_1/batch_normalization_5/AssignNewValue_1�Edecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�4decoder_block_1/batch_normalization_5/ReadVariableOp�6decoder_block_1/batch_normalization_5/ReadVariableOp_1�/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp�.decoder_block_1/conv2d_5/Conv2D/ReadVariableOp�4decoder_block_2/batch_normalization_6/AssignNewValue�6decoder_block_2/batch_normalization_6/AssignNewValue_1�Edecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�4decoder_block_2/batch_normalization_6/ReadVariableOp�6decoder_block_2/batch_normalization_6/ReadVariableOp_1�/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp�.decoder_block_2/conv2d_6/Conv2D/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
dense_2/MatMulMatMulembedding_input%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:������������
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:��*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:�����������b
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*)
_output_shapes
:�����������f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         �
ReshapeReshapedense_2/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:����������d
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:�
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborReshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:����������*
half_pixel_centers(�
,decoder_block/conv2d_4/Conv2D/ReadVariableOpReadVariableOp5decoder_block_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
decoder_block/conv2d_4/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:04decoder_block/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
-decoder_block/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6decoder_block_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
decoder_block/conv2d_4/BiasAddBiasAdd&decoder_block/conv2d_4/Conv2D:output:05decoder_block/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
decoder_block/conv2d_4/ReluRelu'decoder_block/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
2decoder_block/batch_normalization_4/ReadVariableOpReadVariableOp;decoder_block_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
4decoder_block/batch_normalization_4/ReadVariableOp_1ReadVariableOp=decoder_block_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Cdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpLdecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNdecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
4decoder_block/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3)decoder_block/conv2d_4/Relu:activations:0:decoder_block/batch_normalization_4/ReadVariableOp:value:0<decoder_block/batch_normalization_4/ReadVariableOp_1:value:0Kdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Mdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
2decoder_block/batch_normalization_4/AssignNewValueAssignVariableOpLdecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceAdecoder_block/batch_normalization_4/FusedBatchNormV3:batch_mean:0D^decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
4decoder_block/batch_normalization_4/AssignNewValue_1AssignVariableOpNdecoder_block_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceEdecoder_block/batch_normalization_4/FusedBatchNormV3:batch_variance:0F^decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor8decoder_block/batch_normalization_4/FusedBatchNormV3:y:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:���������  �*
half_pixel_centers(�
.decoder_block_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7decoder_block_1_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:�@*
dtype0�
decoder_block_1/conv2d_5/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:06decoder_block_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @*
paddingSAME*
strides
�
/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 decoder_block_1/conv2d_5/BiasAddBiasAdd(decoder_block_1/conv2d_5/Conv2D:output:07decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  @�
decoder_block_1/conv2d_5/ReluRelu)decoder_block_1/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:���������  @�
4decoder_block_1/batch_normalization_5/ReadVariableOpReadVariableOp=decoder_block_1_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
6decoder_block_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp?decoder_block_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Edecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6decoder_block_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3+decoder_block_1/conv2d_5/Relu:activations:0<decoder_block_1/batch_normalization_5/ReadVariableOp:value:0>decoder_block_1/batch_normalization_5/ReadVariableOp_1:value:0Mdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  @:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
4decoder_block_1/batch_normalization_5/AssignNewValueAssignVariableOpNdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceCdecoder_block_1/batch_normalization_5/FusedBatchNormV3:batch_mean:0F^decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6decoder_block_1/batch_normalization_5/AssignNewValue_1AssignVariableOpPdecoder_block_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceGdecoder_block_1/batch_normalization_5/FusedBatchNormV3:batch_variance:0H^decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(f
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      �
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:�
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor:decoder_block_1/batch_normalization_5/FusedBatchNormV3:y:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:�����������@*
half_pixel_centers(�
.decoder_block_2/conv2d_6/Conv2D/ReadVariableOpReadVariableOp7decoder_block_2_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0�
decoder_block_2/conv2d_6/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:06decoder_block_2/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_2_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
 decoder_block_2/conv2d_6/BiasAddBiasAdd(decoder_block_2/conv2d_6/Conv2D:output:07decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� �
decoder_block_2/conv2d_6/ReluRelu)decoder_block_2/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
4decoder_block_2/batch_normalization_6/ReadVariableOpReadVariableOp=decoder_block_2_batch_normalization_6_readvariableop_resource*
_output_shapes
: *
dtype0�
6decoder_block_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp?decoder_block_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes
: *
dtype0�
Edecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
6decoder_block_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3+decoder_block_2/conv2d_6/Relu:activations:0<decoder_block_2/batch_normalization_6/ReadVariableOp:value:0>decoder_block_2/batch_normalization_6/ReadVariableOp_1:value:0Mdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
4decoder_block_2/batch_normalization_6/AssignNewValueAssignVariableOpNdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceCdecoder_block_2/batch_normalization_6/FusedBatchNormV3:batch_mean:0F^decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
6decoder_block_2/batch_normalization_6/AssignNewValue_1AssignVariableOpPdecoder_block_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceGdecoder_block_2/batch_normalization_6/FusedBatchNormV3:batch_variance:0H^decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d_7/Conv2DConv2D:decoder_block_2/batch_normalization_6/FusedBatchNormV3:y:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������l
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:������������
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������r
conv2d_8/SigmoidSigmoidconv2d_8/BiasAdd:output:0*
T0*1
_output_shapes
:�����������m
IdentityIdentityconv2d_8/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp3^decoder_block/batch_normalization_4/AssignNewValue5^decoder_block/batch_normalization_4/AssignNewValue_1D^decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpF^decoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_13^decoder_block/batch_normalization_4/ReadVariableOp5^decoder_block/batch_normalization_4/ReadVariableOp_1.^decoder_block/conv2d_4/BiasAdd/ReadVariableOp-^decoder_block/conv2d_4/Conv2D/ReadVariableOp5^decoder_block_1/batch_normalization_5/AssignNewValue7^decoder_block_1/batch_normalization_5/AssignNewValue_1F^decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpH^decoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_15^decoder_block_1/batch_normalization_5/ReadVariableOp7^decoder_block_1/batch_normalization_5/ReadVariableOp_10^decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp/^decoder_block_1/conv2d_5/Conv2D/ReadVariableOp5^decoder_block_2/batch_normalization_6/AssignNewValue7^decoder_block_2/batch_normalization_6/AssignNewValue_1F^decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^decoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_15^decoder_block_2/batch_normalization_6/ReadVariableOp7^decoder_block_2/batch_normalization_6/ReadVariableOp_10^decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp/^decoder_block_2/conv2d_6/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:����������: : : : : : : : : : : : : : : : : : : : : : : : 2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2l
4decoder_block/batch_normalization_4/AssignNewValue_14decoder_block/batch_normalization_4/AssignNewValue_12h
2decoder_block/batch_normalization_4/AssignNewValue2decoder_block/batch_normalization_4/AssignNewValue2�
Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Edecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12�
Cdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOpCdecoder_block/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2l
4decoder_block/batch_normalization_4/ReadVariableOp_14decoder_block/batch_normalization_4/ReadVariableOp_12h
2decoder_block/batch_normalization_4/ReadVariableOp2decoder_block/batch_normalization_4/ReadVariableOp2^
-decoder_block/conv2d_4/BiasAdd/ReadVariableOp-decoder_block/conv2d_4/BiasAdd/ReadVariableOp2\
,decoder_block/conv2d_4/Conv2D/ReadVariableOp,decoder_block/conv2d_4/Conv2D/ReadVariableOp2p
6decoder_block_1/batch_normalization_5/AssignNewValue_16decoder_block_1/batch_normalization_5/AssignNewValue_12l
4decoder_block_1/batch_normalization_5/AssignNewValue4decoder_block_1/batch_normalization_5/AssignNewValue2�
Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12�
Edecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpEdecoder_block_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_1/batch_normalization_5/ReadVariableOp_16decoder_block_1/batch_normalization_5/ReadVariableOp_12l
4decoder_block_1/batch_normalization_5/ReadVariableOp4decoder_block_1/batch_normalization_5/ReadVariableOp2b
/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp/decoder_block_1/conv2d_5/BiasAdd/ReadVariableOp2`
.decoder_block_1/conv2d_5/Conv2D/ReadVariableOp.decoder_block_1/conv2d_5/Conv2D/ReadVariableOp2p
6decoder_block_2/batch_normalization_6/AssignNewValue_16decoder_block_2/batch_normalization_6/AssignNewValue_12l
4decoder_block_2/batch_normalization_6/AssignNewValue4decoder_block_2/batch_normalization_6/AssignNewValue2�
Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12�
Edecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEdecoder_block_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_2/batch_normalization_6/ReadVariableOp_16decoder_block_2/batch_normalization_6/ReadVariableOp_12l
4decoder_block_2/batch_normalization_6/ReadVariableOp4decoder_block_2/batch_normalization_6/ReadVariableOp2b
/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp/decoder_block_2/conv2d_6/BiasAdd/ReadVariableOp2`
.decoder_block_2/conv2d_6/Conv2D/ReadVariableOp.decoder_block_2/conv2d_6/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
(
_output_shapes
:����������
)
_user_specified_nameembedding_input"�
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

	upsam1


block1

upsam2

block2

upsam3

block3
conv
	final

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 14
!15
"16
#17
$18
%19
&20
'21
(22
)23"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
&14
'15
(16
)17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
/trace_0
0trace_1
1trace_2
2trace_32�
)__inference_decoder_layer_call_fn_1974020
)__inference_decoder_layer_call_fn_1974136
)__inference_decoder_layer_call_fn_1974485
)__inference_decoder_layer_call_fn_1974538�
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
 z/trace_0z0trace_1z1trace_2z2trace_3
�
3trace_0
4trace_1
5trace_2
6trace_32�
D__inference_decoder_layer_call_and_return_conditional_losses_1973765
D__inference_decoder_layer_call_and_return_conditional_losses_1973903
D__inference_decoder_layer_call_and_return_conditional_losses_1974640
D__inference_decoder_layer_call_and_return_conditional_losses_1974742�
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
 z3trace_0z4trace_1z5trace_2z6trace_3
�B�
"__inference__wrapped_model_1973338input_1"�
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
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jbn"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Wconv
Xbn"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
econv
fbn"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

&kernel
'bias
 m_jit_compiled_convolution_op"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

(kernel
)bias
 t_jit_compiled_convolution_op"
_tf_keras_layer
,
userving_default"
signature_map
+:)���2decoder/dense_2/kernel
$:"��2decoder/dense_2/bias
A:?��2%decoder/decoder_block/conv2d_4/kernel
2:0�2#decoder/decoder_block/conv2d_4/bias
@:>�21decoder/decoder_block/batch_normalization_4/gamma
?:=�20decoder/decoder_block/batch_normalization_4/beta
H:F� (27decoder/decoder_block/batch_normalization_4/moving_mean
L:J� (2;decoder/decoder_block/batch_normalization_4/moving_variance
B:@�@2'decoder/decoder_block_1/conv2d_5/kernel
3:1@2%decoder/decoder_block_1/conv2d_5/bias
A:?@23decoder/decoder_block_1/batch_normalization_5/gamma
@:>@22decoder/decoder_block_1/batch_normalization_5/beta
I:G@ (29decoder/decoder_block_1/batch_normalization_5/moving_mean
M:K@ (2=decoder/decoder_block_1/batch_normalization_5/moving_variance
A:?@ 2'decoder/decoder_block_2/conv2d_6/kernel
3:1 2%decoder/decoder_block_2/conv2d_6/bias
A:? 23decoder/decoder_block_2/batch_normalization_6/gamma
@:> 22decoder/decoder_block_2/batch_normalization_6/beta
I:G  (29decoder/decoder_block_2/batch_normalization_6/moving_mean
M:K  (2=decoder/decoder_block_2/batch_normalization_6/moving_variance
1:/ 2decoder/conv2d_7/kernel
#:!2decoder/conv2d_7/bias
1:/2decoder/conv2d_8/kernel
#:!2decoder/conv2d_8/bias
J
0
1
2
3
$4
%5"
trackable_list_wrapper
_
0
	1

2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_decoder_layer_call_fn_1974020input_1"�
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
)__inference_decoder_layer_call_fn_1974136input_1"�
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
)__inference_decoder_layer_call_fn_1974485embedding_input"�
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
)__inference_decoder_layer_call_fn_1974538embedding_input"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1973765input_1"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1973903input_1"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1974640embedding_input"�
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
D__inference_decoder_layer_call_and_return_conditional_losses_1974742embedding_input"�
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
{trace_02�
)__inference_dense_2_layer_call_fn_1974751�
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
 z{trace_0
�
|trace_02�
D__inference_dense_2_layer_call_and_return_conditional_losses_1974762�
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
 z|trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_up_sampling2d_layer_call_fn_1974767�
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1974779�
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
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
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
/__inference_decoder_block_layer_call_fn_1974796
/__inference_decoder_block_layer_call_fn_1974813�
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
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974838
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974863�
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

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
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
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_1_layer_call_fn_1974868�
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1974880�
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
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_decoder_block_1_layer_call_fn_1974897
1__inference_decoder_block_1_layer_call_fn_1974914�
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
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974939
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974964�
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

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
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
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_up_sampling2d_2_layer_call_fn_1974969�
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
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1974981�
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
J
 0
!1
"2
#3
$4
%5"
trackable_list_wrapper
<
 0
!1
"2
#3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_decoder_block_2_layer_call_fn_1974998
1__inference_decoder_block_2_layer_call_fn_1975015�
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
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975040
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975065�
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

 kernel
!bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	"gamma
#beta
$moving_mean
%moving_variance"
_tf_keras_layer
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_7_layer_call_fn_1975074�
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
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1975085�
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
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_8_layer_call_fn_1975094�
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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1975105�
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
%__inference_signature_wrapper_1974432input_1"�
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
)__inference_dense_2_layer_call_fn_1974751inputs"�
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
D__inference_dense_2_layer_call_and_return_conditional_losses_1974762inputs"�
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
/__inference_up_sampling2d_layer_call_fn_1974767inputs"�
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1974779inputs"�
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
0
1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_decoder_block_layer_call_fn_1974796input_tensor"�
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
/__inference_decoder_block_layer_call_fn_1974813input_tensor"�
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
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974838input_tensor"�
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
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974863input_tensor"�
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
0
1"
trackable_list_wrapper
.
0
1"
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
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
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
7__inference_batch_normalization_4_layer_call_fn_1975118
7__inference_batch_normalization_4_layer_call_fn_1975131�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975149
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975167�
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
1__inference_up_sampling2d_1_layer_call_fn_1974868inputs"�
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1974880inputs"�
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
0
1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_decoder_block_1_layer_call_fn_1974897input_tensor"�
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
1__inference_decoder_block_1_layer_call_fn_1974914input_tensor"�
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
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974939input_tensor"�
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
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974964input_tensor"�
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
0
1"
trackable_list_wrapper
.
0
1"
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
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
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
7__inference_batch_normalization_5_layer_call_fn_1975180
7__inference_batch_normalization_5_layer_call_fn_1975193�
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
�trace_12�
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975211
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975229�
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
 z�trace_0z�trace_1
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
1__inference_up_sampling2d_2_layer_call_fn_1974969inputs"�
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
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1974981inputs"�
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
$0
%1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_decoder_block_2_layer_call_fn_1974998input_tensor"�
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
1__inference_decoder_block_2_layer_call_fn_1975015input_tensor"�
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
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975040input_tensor"�
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
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975065input_tensor"�
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
 0
!1"
trackable_list_wrapper
.
 0
!1"
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
<
"0
#1
$2
%3"
trackable_list_wrapper
.
"0
#1"
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
7__inference_batch_normalization_6_layer_call_fn_1975242
7__inference_batch_normalization_6_layer_call_fn_1975255�
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975273
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975291�
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
*__inference_conv2d_7_layer_call_fn_1975074inputs"�
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
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1975085inputs"�
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
*__inference_conv2d_8_layer_call_fn_1975094inputs"�
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
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1975105inputs"�
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
.
0
1"
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
7__inference_batch_normalization_4_layer_call_fn_1975118inputs"�
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
7__inference_batch_normalization_4_layer_call_fn_1975131inputs"�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975149inputs"�
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975167inputs"�
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
.
0
1"
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
7__inference_batch_normalization_5_layer_call_fn_1975180inputs"�
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
7__inference_batch_normalization_5_layer_call_fn_1975193inputs"�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975211inputs"�
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975229inputs"�
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
.
$0
%1"
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
7__inference_batch_normalization_6_layer_call_fn_1975242inputs"�
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
7__inference_batch_normalization_6_layer_call_fn_1975255inputs"�
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975273inputs"�
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975291inputs"�
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
"__inference__wrapped_model_1973338� !"#$%&'()1�.
'�$
"�
input_1����������
� "=�:
8
output_1,�)
output_1������������
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975149�R�O
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
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1975167�R�O
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
7__inference_batch_normalization_4_layer_call_fn_1975118�R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
7__inference_batch_normalization_4_layer_call_fn_1975131�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975211�Q�N
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
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1975229�Q�N
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
7__inference_batch_normalization_5_layer_call_fn_1975180�Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
7__inference_batch_normalization_5_layer_call_fn_1975193�Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975273�"#$%Q�N
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
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1975291�"#$%Q�N
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
7__inference_batch_normalization_6_layer_call_fn_1975242�"#$%Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
7__inference_batch_normalization_6_layer_call_fn_1975255�"#$%Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
E__inference_conv2d_7_layer_call_and_return_conditional_losses_1975085�&'I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������
� �
*__inference_conv2d_7_layer_call_fn_1975074�&'I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+����������������������������
E__inference_conv2d_8_layer_call_and_return_conditional_losses_1975105�()I�F
?�<
:�7
inputs+���������������������������
� "F�C
<�9
tensor_0+���������������������������
� �
*__inference_conv2d_8_layer_call_fn_1975094�()I�F
?�<
:�7
inputs+���������������������������
� ";�8
unknown+����������������������������
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974939�T�Q
J�G
A�>
input_tensor,����������������������������
p
� "F�C
<�9
tensor_0+���������������������������@
� �
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1974964�T�Q
J�G
A�>
input_tensor,����������������������������
p 
� "F�C
<�9
tensor_0+���������������������������@
� �
1__inference_decoder_block_1_layer_call_fn_1974897�T�Q
J�G
A�>
input_tensor,����������������������������
p
� ";�8
unknown+���������������������������@�
1__inference_decoder_block_1_layer_call_fn_1974914�T�Q
J�G
A�>
input_tensor,����������������������������
p 
� ";�8
unknown+���������������������������@�
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975040� !"#$%S�P
I�F
@�=
input_tensor+���������������������������@
p
� "F�C
<�9
tensor_0+��������������������������� 
� �
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1975065� !"#$%S�P
I�F
@�=
input_tensor+���������������������������@
p 
� "F�C
<�9
tensor_0+��������������������������� 
� �
1__inference_decoder_block_2_layer_call_fn_1974998� !"#$%S�P
I�F
@�=
input_tensor+���������������������������@
p
� ";�8
unknown+��������������������������� �
1__inference_decoder_block_2_layer_call_fn_1975015� !"#$%S�P
I�F
@�=
input_tensor+���������������������������@
p 
� ";�8
unknown+��������������������������� �
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974838�T�Q
J�G
A�>
input_tensor,����������������������������
p
� "G�D
=�:
tensor_0,����������������������������
� �
J__inference_decoder_block_layer_call_and_return_conditional_losses_1974863�T�Q
J�G
A�>
input_tensor,����������������������������
p 
� "G�D
=�:
tensor_0,����������������������������
� �
/__inference_decoder_block_layer_call_fn_1974796�T�Q
J�G
A�>
input_tensor,����������������������������
p
� "<�9
unknown,�����������������������������
/__inference_decoder_block_layer_call_fn_1974813�T�Q
J�G
A�>
input_tensor,����������������������������
p 
� "<�9
unknown,�����������������������������
D__inference_decoder_layer_call_and_return_conditional_losses_1973765� !"#$%&'()A�>
'�$
"�
input_1����������
�

trainingp"F�C
<�9
tensor_0+���������������������������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1973903� !"#$%&'()A�>
'�$
"�
input_1����������
�

trainingp "F�C
<�9
tensor_0+���������������������������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1974640� !"#$%&'()I�F
/�,
*�'
embedding_input����������
�

trainingp"6�3
,�)
tensor_0�����������
� �
D__inference_decoder_layer_call_and_return_conditional_losses_1974742� !"#$%&'()I�F
/�,
*�'
embedding_input����������
�

trainingp "6�3
,�)
tensor_0�����������
� �
)__inference_decoder_layer_call_fn_1974020� !"#$%&'()A�>
'�$
"�
input_1����������
�

trainingp";�8
unknown+����������������������������
)__inference_decoder_layer_call_fn_1974136� !"#$%&'()A�>
'�$
"�
input_1����������
�

trainingp ";�8
unknown+����������������������������
)__inference_decoder_layer_call_fn_1974485� !"#$%&'()I�F
/�,
*�'
embedding_input����������
�

trainingp";�8
unknown+����������������������������
)__inference_decoder_layer_call_fn_1974538� !"#$%&'()I�F
/�,
*�'
embedding_input����������
�

trainingp ";�8
unknown+����������������������������
D__inference_dense_2_layer_call_and_return_conditional_losses_1974762f0�-
&�#
!�
inputs����������
� ".�+
$�!
tensor_0�����������
� �
)__inference_dense_2_layer_call_fn_1974751[0�-
&�#
!�
inputs����������
� "#� 
unknown������������
%__inference_signature_wrapper_1974432� !"#$%&'()<�9
� 
2�/
-
input_1"�
input_1����������"=�:
8
output_1,�)
output_1������������
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1974880�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_1_layer_call_fn_1974868�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1974981�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_up_sampling2d_2_layer_call_fn_1974969�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1974779�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
/__inference_up_sampling2d_layer_call_fn_1974767�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4������������������������������������