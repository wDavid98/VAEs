ба
фщ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
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
Џ
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
resourceѕ
ч
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
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
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
Ў
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758 Ћ
ё
decoder/conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder/conv2d_10/bias
}
*decoder/conv2d_10/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_10/bias*
_output_shapes
:*
dtype0
ћ
decoder/conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namedecoder/conv2d_10/kernel
Ї
,decoder/conv2d_10/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_10/kernel*&
_output_shapes
:*
dtype0
ѓ
decoder/conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedecoder/conv2d_9/bias
{
)decoder/conv2d_9/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_9/bias*
_output_shapes
:*
dtype0
њ
decoder/conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namedecoder/conv2d_9/kernel
І
+decoder/conv2d_9/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_9/kernel*&
_output_shapes
: *
dtype0
м
=decoder/decoder_block_2/batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=decoder/decoder_block_2/batch_normalization_7/moving_variance
╦
Qdecoder/decoder_block_2/batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp=decoder/decoder_block_2/batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
╩
9decoder/decoder_block_2/batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9decoder/decoder_block_2/batch_normalization_7/moving_mean
├
Mdecoder/decoder_block_2/batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp9decoder/decoder_block_2/batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
╝
2decoder/decoder_block_2/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42decoder/decoder_block_2/batch_normalization_7/beta
х
Fdecoder/decoder_block_2/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp2decoder/decoder_block_2/batch_normalization_7/beta*
_output_shapes
: *
dtype0
Й
3decoder/decoder_block_2/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53decoder/decoder_block_2/batch_normalization_7/gamma
и
Gdecoder/decoder_block_2/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_2/batch_normalization_7/gamma*
_output_shapes
: *
dtype0
б
%decoder/decoder_block_2/conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%decoder/decoder_block_2/conv2d_8/bias
Џ
9decoder/decoder_block_2/conv2d_8/bias/Read/ReadVariableOpReadVariableOp%decoder/decoder_block_2/conv2d_8/bias*
_output_shapes
: *
dtype0
▓
'decoder/decoder_block_2/conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *8
shared_name)'decoder/decoder_block_2/conv2d_8/kernel
Ф
;decoder/decoder_block_2/conv2d_8/kernel/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_2/conv2d_8/kernel*&
_output_shapes
:@ *
dtype0
м
=decoder/decoder_block_1/batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*N
shared_name?=decoder/decoder_block_1/batch_normalization_6/moving_variance
╦
Qdecoder/decoder_block_1/batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp=decoder/decoder_block_1/batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
╩
9decoder/decoder_block_1/batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*J
shared_name;9decoder/decoder_block_1/batch_normalization_6/moving_mean
├
Mdecoder/decoder_block_1/batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp9decoder/decoder_block_1/batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
╝
2decoder/decoder_block_1/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42decoder/decoder_block_1/batch_normalization_6/beta
х
Fdecoder/decoder_block_1/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp2decoder/decoder_block_1/batch_normalization_6/beta*
_output_shapes
:@*
dtype0
Й
3decoder/decoder_block_1/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53decoder/decoder_block_1/batch_normalization_6/gamma
и
Gdecoder/decoder_block_1/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp3decoder/decoder_block_1/batch_normalization_6/gamma*
_output_shapes
:@*
dtype0
б
%decoder/decoder_block_1/conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%decoder/decoder_block_1/conv2d_7/bias
Џ
9decoder/decoder_block_1/conv2d_7/bias/Read/ReadVariableOpReadVariableOp%decoder/decoder_block_1/conv2d_7/bias*
_output_shapes
:@*
dtype0
│
'decoder/decoder_block_1/conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ@*8
shared_name)'decoder/decoder_block_1/conv2d_7/kernel
г
;decoder/decoder_block_1/conv2d_7/kernel/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_1/conv2d_7/kernel*'
_output_shapes
:ђ@*
dtype0
¤
;decoder/decoder_block/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;decoder/decoder_block/batch_normalization_5/moving_variance
╚
Odecoder/decoder_block/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp;decoder/decoder_block/batch_normalization_5/moving_variance*
_output_shapes	
:ђ*
dtype0
К
7decoder/decoder_block/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*H
shared_name97decoder/decoder_block/batch_normalization_5/moving_mean
└
Kdecoder/decoder_block/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp7decoder/decoder_block/batch_normalization_5/moving_mean*
_output_shapes	
:ђ*
dtype0
╣
0decoder/decoder_block/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*A
shared_name20decoder/decoder_block/batch_normalization_5/beta
▓
Ddecoder/decoder_block/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp0decoder/decoder_block/batch_normalization_5/beta*
_output_shapes	
:ђ*
dtype0
╗
1decoder/decoder_block/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*B
shared_name31decoder/decoder_block/batch_normalization_5/gamma
┤
Edecoder/decoder_block/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp1decoder/decoder_block/batch_normalization_5/gamma*
_output_shapes	
:ђ*
dtype0
Ъ
#decoder/decoder_block/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*4
shared_name%#decoder/decoder_block/conv2d_6/bias
ў
7decoder/decoder_block/conv2d_6/bias/Read/ReadVariableOpReadVariableOp#decoder/decoder_block/conv2d_6/bias*
_output_shapes	
:ђ*
dtype0
░
%decoder/decoder_block/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*6
shared_name'%decoder/decoder_block/conv2d_6/kernel
Е
9decoder/decoder_block/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp%decoder/decoder_block/conv2d_6/kernel*(
_output_shapes
:ђђ*
dtype0
ѓ
decoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*%
shared_namedecoder/dense_2/bias
{
(decoder/dense_2/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_2/bias*
_output_shapes

:ђђ*
dtype0
І
decoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђђ*'
shared_namedecoder/dense_2/kernel
ё
*decoder/dense_2/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_2/kernel*!
_output_shapes
:ђђђ*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
Ј
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1decoder/dense_2/kerneldecoder/dense_2/bias%decoder/decoder_block/conv2d_6/kernel#decoder/decoder_block/conv2d_6/bias1decoder/decoder_block/batch_normalization_5/gamma0decoder/decoder_block/batch_normalization_5/beta7decoder/decoder_block/batch_normalization_5/moving_mean;decoder/decoder_block/batch_normalization_5/moving_variance'decoder/decoder_block_1/conv2d_7/kernel%decoder/decoder_block_1/conv2d_7/bias3decoder/decoder_block_1/batch_normalization_6/gamma2decoder/decoder_block_1/batch_normalization_6/beta9decoder/decoder_block_1/batch_normalization_6/moving_mean=decoder/decoder_block_1/batch_normalization_6/moving_variance'decoder/decoder_block_2/conv2d_8/kernel%decoder/decoder_block_2/conv2d_8/bias3decoder/decoder_block_2/batch_normalization_7/gamma2decoder/decoder_block_2/batch_normalization_7/beta9decoder/decoder_block_2/batch_normalization_7/moving_mean=decoder/decoder_block_2/batch_normalization_7/moving_variancedecoder/conv2d_9/kerneldecoder/conv2d_9/biasdecoder/conv2d_10/kerneldecoder/conv2d_10/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1905060

NoOpNoOp
┐e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Щd
value­dBьd BТd
Є
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
║
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
і
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
░
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
д
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

kernel
bias*
Г
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@conv
	Aupsam
Bbn*
Г
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
	Jupsam
Kbn*
Г
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rconv
	Supsam
Tbn*
╚
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

#kernel
$bias
 [_jit_compiled_convolution_op*
╚
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
VARIABLE_VALUEdecoder/dense_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdecoder/dense_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%decoder/decoder_block/conv2d_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#decoder/decoder_block/conv2d_6/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1decoder/decoder_block/batch_normalization_5/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0decoder/decoder_block/batch_normalization_5/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE7decoder/decoder_block/batch_normalization_5/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;decoder/decoder_block/batch_normalization_5/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'decoder/decoder_block_1/conv2d_7/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%decoder/decoder_block_1/conv2d_7/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3decoder/decoder_block_1/batch_normalization_6/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2decoder/decoder_block_1/batch_normalization_6/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9decoder/decoder_block_1/batch_normalization_6/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=decoder/decoder_block_1/batch_normalization_6/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'decoder/decoder_block_2/conv2d_8/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%decoder/decoder_block_2/conv2d_8/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE3decoder/decoder_block_2/batch_normalization_7/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2decoder/decoder_block_2/batch_normalization_7/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE9decoder/decoder_block_2/batch_normalization_7/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE=decoder/decoder_block_2/batch_normalization_7/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEdecoder/conv2d_9/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdecoder/conv2d_9/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_10/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_10/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
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
Њ
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
Њ
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
╚
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

kernel
bias
 z_jit_compiled_convolution_op*
Ј
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+ђ&call_and_return_all_conditional_losses* 
▄
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
ё	keras_api
Ё__call__
+є&call_and_return_all_conditional_losses
	Єaxis
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
ў
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

Їtrace_0
јtrace_1* 

Јtrace_0
љtrace_1* 
¤
Љ	variables
њtrainable_variables
Њregularization_losses
ћ	keras_api
Ћ__call__
+ќ&call_and_return_all_conditional_losses

kernel
bias
!Ќ_jit_compiled_convolution_op*
ћ
ў	variables
Ўtrainable_variables
џregularization_losses
Џ	keras_api
ю__call__
+Ю&call_and_return_all_conditional_losses* 
▄
ъ	variables
Ъtrainable_variables
аregularization_losses
А	keras_api
б__call__
+Б&call_and_return_all_conditional_losses
	цaxis
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
ў
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

фtrace_0
Фtrace_1* 

гtrace_0
Гtrace_1* 
¤
«	variables
»trainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses

kernel
bias
!┤_jit_compiled_convolution_op*
ћ
х	variables
Хtrainable_variables
иregularization_losses
И	keras_api
╣__call__
+║&call_and_return_all_conditional_losses* 
▄
╗	variables
╝trainable_variables
йregularization_losses
Й	keras_api
┐__call__
+└&call_and_return_all_conditional_losses
	┴axis
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
ў
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
кlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

Кtrace_0* 

╚trace_0* 
* 

%0
&1*

%0
&1*
* 
ў
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

╬trace_0* 

¤trace_0* 
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
ў
лnon_trainable_variables
Лlayers
мmetrics
 Мlayer_regularization_losses
нlayer_metrics
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
ў
Нnon_trainable_variables
оlayers
Оmetrics
 пlayer_regularization_losses
┘layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses* 

┌trace_0* 

█trace_0* 
 
0
1
2
3*

0
1*
* 
ъ
▄non_trainable_variables
Пlayers
яmetrics
 ▀layer_regularization_losses
Яlayer_metrics
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
Ё__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses*

рtrace_0
Рtrace_1* 

сtrace_0
Сtrace_1* 
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
ъ
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
Љ	variables
њtrainable_variables
Њregularization_losses
Ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
ў	variables
Ўtrainable_variables
џregularization_losses
ю__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses* 

№trace_0* 

­trace_0* 
 
0
1
2
3*

0
1*
* 
ъ
ыnon_trainable_variables
Ыlayers
зmetrics
 Зlayer_regularization_losses
шlayer_metrics
ъ	variables
Ъtrainable_variables
аregularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

Шtrace_0
эtrace_1* 

Эtrace_0
щtrace_1* 
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
ъ
Щnon_trainable_variables
чlayers
Чmetrics
 §layer_regularization_losses
■layer_metrics
«	variables
»trainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
 non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
х	variables
Хtrainable_variables
иregularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses* 

ёtrace_0* 

Ёtrace_0* 
 
0
 1
!2
"3*

0
 1*
* 
ъ
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
╗	variables
╝trainable_variables
йregularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses*

Іtrace_0
їtrace_1* 

Їtrace_0
јtrace_1* 
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
о

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedecoder/dense_2/kerneldecoder/dense_2/bias%decoder/decoder_block/conv2d_6/kernel#decoder/decoder_block/conv2d_6/bias1decoder/decoder_block/batch_normalization_5/gamma0decoder/decoder_block/batch_normalization_5/beta7decoder/decoder_block/batch_normalization_5/moving_mean;decoder/decoder_block/batch_normalization_5/moving_variance'decoder/decoder_block_1/conv2d_7/kernel%decoder/decoder_block_1/conv2d_7/bias3decoder/decoder_block_1/batch_normalization_6/gamma2decoder/decoder_block_1/batch_normalization_6/beta9decoder/decoder_block_1/batch_normalization_6/moving_mean=decoder/decoder_block_1/batch_normalization_6/moving_variance'decoder/decoder_block_2/conv2d_8/kernel%decoder/decoder_block_2/conv2d_8/bias3decoder/decoder_block_2/batch_normalization_7/gamma2decoder/decoder_block_2/batch_normalization_7/beta9decoder/decoder_block_2/batch_normalization_7/moving_mean=decoder/decoder_block_2/batch_normalization_7/moving_variancedecoder/conv2d_9/kerneldecoder/conv2d_9/biasdecoder/conv2d_10/kerneldecoder/conv2d_10/biasConst*%
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
GPU2 *0J 8ѓ *)
f$R"
 __inference__traced_save_1906110
Л

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder/dense_2/kerneldecoder/dense_2/bias%decoder/decoder_block/conv2d_6/kernel#decoder/decoder_block/conv2d_6/bias1decoder/decoder_block/batch_normalization_5/gamma0decoder/decoder_block/batch_normalization_5/beta7decoder/decoder_block/batch_normalization_5/moving_mean;decoder/decoder_block/batch_normalization_5/moving_variance'decoder/decoder_block_1/conv2d_7/kernel%decoder/decoder_block_1/conv2d_7/bias3decoder/decoder_block_1/batch_normalization_6/gamma2decoder/decoder_block_1/batch_normalization_6/beta9decoder/decoder_block_1/batch_normalization_6/moving_mean=decoder/decoder_block_1/batch_normalization_6/moving_variance'decoder/decoder_block_2/conv2d_8/kernel%decoder/decoder_block_2/conv2d_8/bias3decoder/decoder_block_2/batch_normalization_7/gamma2decoder/decoder_block_2/batch_normalization_7/beta9decoder/decoder_block_2/batch_normalization_7/moving_mean=decoder/decoder_block_2/batch_normalization_7/moving_variancedecoder/conv2d_9/kerneldecoder/conv2d_9/biasdecoder/conv2d_10/kerneldecoder/conv2d_10/bias*$
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
GPU2 *0J 8ѓ *,
f'R%
#__inference__traced_restore_1906192с╠
б
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1905802

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:й
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
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
б
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1904133

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:й
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
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
О)
█
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904341
input_tensorA
'conv2d_8_conv2d_readvariableop_resource:@ 6
(conv2d_8_biasadd_readvariableop_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1бconv2d_8/BiasAdd/ReadVariableOpбconv2d_8/Conv2D/ReadVariableOpј
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0▒
conv2d_8/Conv2DConv2Dinput_tensor&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
ё
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            f
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Л
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_8/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(ј
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0в
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ Џ
NoOpNoOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':           @: : : : : : 2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:           @
&
_user_specified_nameinput_tensor
»

Щ
D__inference_dense_2_layer_call_and_return_conditional_losses_1905390

inputs3
matmul_readvariableop_resource:ђђђ/
biasadd_readvariableop_resource:
ђђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђt
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђR
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:         ђђc
IdentityIdentityRelu:activations:0^NoOp*
T0*)
_output_shapes
:         ђђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
│	
љ
1__inference_decoder_block_2_layer_call_fn_1905591
input_tensor!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904341y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':           @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:           @
&
_user_specified_nameinput_tensor
Н)
▄
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905545
input_tensorB
'conv2d_7_conv2d_readvariableop_resource:ђ@6
(conv2d_7_biasadd_readvariableop_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_6/AssignNewValueб&batch_normalization_6/AssignNewValue_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1бconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpЈ
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0▒
conv2d_7/Conv2DConv2Dinput_tensor&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:¤
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_7/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(ј
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ђ
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:           @Џ
NoOpNoOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
▄Г
ь
D__inference_decoder_layer_call_and_return_conditional_losses_1905268
embedding_input;
&dense_2_matmul_readvariableop_resource:ђђђ7
'dense_2_biasadd_readvariableop_resource:
ђђQ
5decoder_block_conv2d_6_conv2d_readvariableop_resource:ђђE
6decoder_block_conv2d_6_biasadd_readvariableop_resource:	ђJ
;decoder_block_batch_normalization_5_readvariableop_resource:	ђL
=decoder_block_batch_normalization_5_readvariableop_1_resource:	ђ[
Ldecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђ]
Ndecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђR
7decoder_block_1_conv2d_7_conv2d_readvariableop_resource:ђ@F
8decoder_block_1_conv2d_7_biasadd_readvariableop_resource:@K
=decoder_block_1_batch_normalization_6_readvariableop_resource:@M
?decoder_block_1_batch_normalization_6_readvariableop_1_resource:@\
Ndecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@^
Pdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@Q
7decoder_block_2_conv2d_8_conv2d_readvariableop_resource:@ F
8decoder_block_2_conv2d_8_biasadd_readvariableop_resource: K
=decoder_block_2_batch_normalization_7_readvariableop_resource: M
?decoder_block_2_batch_normalization_7_readvariableop_1_resource: \
Ndecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: ^
Pdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:
identityѕб conv2d_10/BiasAdd/ReadVariableOpбconv2d_10/Conv2D/ReadVariableOpбconv2d_9/BiasAdd/ReadVariableOpбconv2d_9/Conv2D/ReadVariableOpб2decoder_block/batch_normalization_5/AssignNewValueб4decoder_block/batch_normalization_5/AssignNewValue_1бCdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбEdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б2decoder_block/batch_normalization_5/ReadVariableOpб4decoder_block/batch_normalization_5/ReadVariableOp_1б-decoder_block/conv2d_6/BiasAdd/ReadVariableOpб,decoder_block/conv2d_6/Conv2D/ReadVariableOpб4decoder_block_1/batch_normalization_6/AssignNewValueб6decoder_block_1/batch_normalization_6/AssignNewValue_1бEdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpбGdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б4decoder_block_1/batch_normalization_6/ReadVariableOpб6decoder_block_1/batch_normalization_6/ReadVariableOp_1б/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOpб.decoder_block_1/conv2d_7/Conv2D/ReadVariableOpб4decoder_block_2/batch_normalization_7/AssignNewValueб6decoder_block_2/batch_normalization_7/AssignNewValue_1бEdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбGdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б4decoder_block_2/batch_normalization_7/ReadVariableOpб6decoder_block_2/batch_normalization_7/ReadVariableOp_1б/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOpб.decoder_block_2/conv2d_8/Conv2D/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpЄ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0ё
dense_2/MatMulMatMulembedding_input%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђё
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0љ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђb
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*)
_output_shapes
:         ђђf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ђ
ReshapeReshapedense_2/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђг
,decoder_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp5decoder_block_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0м
decoder_block/conv2d_6/Conv2DConv2DReshape:output:04decoder_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
А
-decoder_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6decoder_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0├
decoder_block/conv2d_6/BiasAddBiasAdd&decoder_block/conv2d_6/Conv2D:output:05decoder_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЄ
decoder_block/conv2d_6/ReluRelu'decoder_block/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђr
!decoder_block/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      t
#decoder_block/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ц
decoder_block/up_sampling2d/mulMul*decoder_block/up_sampling2d/Const:output:0,decoder_block/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:Ш
8decoder_block/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor)decoder_block/conv2d_6/Relu:activations:0#decoder_block/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Ф
2decoder_block/batch_normalization_5/ReadVariableOpReadVariableOp;decoder_block_batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0»
4decoder_block/batch_normalization_5/ReadVariableOp_1ReadVariableOp=decoder_block_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
Cdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpLdecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Л
Edecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNdecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0└
4decoder_block/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3Idecoder_block/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:decoder_block/batch_normalization_5/ReadVariableOp:value:0<decoder_block/batch_normalization_5/ReadVariableOp_1:value:0Kdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Mdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<о
2decoder_block/batch_normalization_5/AssignNewValueAssignVariableOpLdecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceAdecoder_block/batch_normalization_5/FusedBatchNormV3:batch_mean:0D^decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Я
4decoder_block/batch_normalization_5/AssignNewValue_1AssignVariableOpNdecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceEdecoder_block/batch_normalization_5/FusedBatchNormV3:batch_variance:0F^decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(»
.decoder_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp7decoder_block_1_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0§
decoder_block_1/conv2d_7/Conv2DConv2D8decoder_block/batch_normalization_5/FusedBatchNormV3:y:06decoder_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ц
/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╚
 decoder_block_1/conv2d_7/BiasAddBiasAdd(decoder_block_1/conv2d_7/Conv2D:output:07decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @і
decoder_block_1/conv2d_7/ReluRelu)decoder_block_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @v
%decoder_block_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'decoder_block_1/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ▒
#decoder_block_1/up_sampling2d_1/mulMul.decoder_block_1/up_sampling2d_1/Const:output:00decoder_block_1/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
: 
<decoder_block_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor+decoder_block_1/conv2d_7/Relu:activations:0'decoder_block_1/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(«
4decoder_block_1/batch_normalization_6/ReadVariableOpReadVariableOp=decoder_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0▓
6decoder_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp?decoder_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0л
Edecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╔
6decoder_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0<decoder_block_1/batch_normalization_6/ReadVariableOp:value:0>decoder_block_1/batch_normalization_6/ReadVariableOp_1:value:0Mdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<я
4decoder_block_1/batch_normalization_6/AssignNewValueAssignVariableOpNdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceCdecoder_block_1/batch_normalization_6/FusedBatchNormV3:batch_mean:0F^decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(У
6decoder_block_1/batch_normalization_6/AssignNewValue_1AssignVariableOpPdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceGdecoder_block_1/batch_normalization_6/FusedBatchNormV3:batch_variance:0H^decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(«
.decoder_block_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp7decoder_block_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0 
decoder_block_2/conv2d_8/Conv2DConv2D:decoder_block_1/batch_normalization_6/FusedBatchNormV3:y:06decoder_block_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
ц
/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╚
 decoder_block_2/conv2d_8/BiasAddBiasAdd(decoder_block_2/conv2d_8/Conv2D:output:07decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            і
decoder_block_2/conv2d_8/ReluRelu)decoder_block_2/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            v
%decoder_block_2/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        x
'decoder_block_2/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ▒
#decoder_block_2/up_sampling2d_2/mulMul.decoder_block_2/up_sampling2d_2/Const:output:00decoder_block_2/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Ђ
<decoder_block_2/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor+decoder_block_2/conv2d_8/Relu:activations:0'decoder_block_2/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(«
4decoder_block_2/batch_normalization_7/ReadVariableOpReadVariableOp=decoder_block_2_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0▓
6decoder_block_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp?decoder_block_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0л
Edecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0н
Gdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╦
6decoder_block_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_2/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0<decoder_block_2/batch_normalization_7/ReadVariableOp:value:0>decoder_block_2/batch_normalization_7/ReadVariableOp_1:value:0Mdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<я
4decoder_block_2/batch_normalization_7/AssignNewValueAssignVariableOpNdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceCdecoder_block_2/batch_normalization_7/FusedBatchNormV3:batch_mean:0F^decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(У
6decoder_block_2/batch_normalization_7/AssignNewValue_1AssignVariableOpPdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceGdecoder_block_2/batch_normalization_7/FusedBatchNormV3:batch_variance:0H^decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ј
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0р
conv2d_9/Conv2DConv2D:decoder_block_2/batch_normalization_7/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
ё
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0џ
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђl
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђљ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
є
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђt
conv2d_10/SigmoidSigmoidconv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђn
IdentityIdentityconv2d_10/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђ■
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp3^decoder_block/batch_normalization_5/AssignNewValue5^decoder_block/batch_normalization_5/AssignNewValue_1D^decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpF^decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_13^decoder_block/batch_normalization_5/ReadVariableOp5^decoder_block/batch_normalization_5/ReadVariableOp_1.^decoder_block/conv2d_6/BiasAdd/ReadVariableOp-^decoder_block/conv2d_6/Conv2D/ReadVariableOp5^decoder_block_1/batch_normalization_6/AssignNewValue7^decoder_block_1/batch_normalization_6/AssignNewValue_1F^decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_15^decoder_block_1/batch_normalization_6/ReadVariableOp7^decoder_block_1/batch_normalization_6/ReadVariableOp_10^decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp/^decoder_block_1/conv2d_7/Conv2D/ReadVariableOp5^decoder_block_2/batch_normalization_7/AssignNewValue7^decoder_block_2/batch_normalization_7/AssignNewValue_1F^decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpH^decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_15^decoder_block_2/batch_normalization_7/ReadVariableOp7^decoder_block_2/batch_normalization_7/ReadVariableOp_10^decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp/^decoder_block_2/conv2d_8/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2l
4decoder_block/batch_normalization_5/AssignNewValue_14decoder_block/batch_normalization_5/AssignNewValue_12h
2decoder_block/batch_normalization_5/AssignNewValue2decoder_block/batch_normalization_5/AssignNewValue2ј
Edecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Edecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12і
Cdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpCdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2l
4decoder_block/batch_normalization_5/ReadVariableOp_14decoder_block/batch_normalization_5/ReadVariableOp_12h
2decoder_block/batch_normalization_5/ReadVariableOp2decoder_block/batch_normalization_5/ReadVariableOp2^
-decoder_block/conv2d_6/BiasAdd/ReadVariableOp-decoder_block/conv2d_6/BiasAdd/ReadVariableOp2\
,decoder_block/conv2d_6/Conv2D/ReadVariableOp,decoder_block/conv2d_6/Conv2D/ReadVariableOp2p
6decoder_block_1/batch_normalization_6/AssignNewValue_16decoder_block_1/batch_normalization_6/AssignNewValue_12l
4decoder_block_1/batch_normalization_6/AssignNewValue4decoder_block_1/batch_normalization_6/AssignNewValue2њ
Gdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12ј
Edecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_1/batch_normalization_6/ReadVariableOp_16decoder_block_1/batch_normalization_6/ReadVariableOp_12l
4decoder_block_1/batch_normalization_6/ReadVariableOp4decoder_block_1/batch_normalization_6/ReadVariableOp2b
/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp2`
.decoder_block_1/conv2d_7/Conv2D/ReadVariableOp.decoder_block_1/conv2d_7/Conv2D/ReadVariableOp2p
6decoder_block_2/batch_normalization_7/AssignNewValue_16decoder_block_2/batch_normalization_7/AssignNewValue_12l
4decoder_block_2/batch_normalization_7/AssignNewValue4decoder_block_2/batch_normalization_7/AssignNewValue2њ
Gdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12ј
Edecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpEdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_2/batch_normalization_7/ReadVariableOp_16decoder_block_2/batch_normalization_7/ReadVariableOp_12l
4decoder_block_2/batch_normalization_7/ReadVariableOp4decoder_block_2/batch_normalization_7/ReadVariableOp2b
/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp2`
.decoder_block_2/conv2d_8/Conv2D/ReadVariableOp.decoder_block_2/conv2d_8/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
Х	
Ћ
/__inference_decoder_block_layer_call_fn_1905407
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904255x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
┐
M
1__inference_up_sampling2d_2_layer_call_fn_1905869

inputs
identity▀
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1904133Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905925

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
а
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1905723

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:й
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
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ч┐
В
 __inference__traced_save_1906110
file_prefixB
-read_disablecopyonread_decoder_dense_2_kernel:ђђђ=
-read_1_disablecopyonread_decoder_dense_2_bias:
ђђZ
>read_2_disablecopyonread_decoder_decoder_block_conv2d_6_kernel:ђђK
<read_3_disablecopyonread_decoder_decoder_block_conv2d_6_bias:	ђY
Jread_4_disablecopyonread_decoder_decoder_block_batch_normalization_5_gamma:	ђX
Iread_5_disablecopyonread_decoder_decoder_block_batch_normalization_5_beta:	ђ_
Pread_6_disablecopyonread_decoder_decoder_block_batch_normalization_5_moving_mean:	ђc
Tread_7_disablecopyonread_decoder_decoder_block_batch_normalization_5_moving_variance:	ђ[
@read_8_disablecopyonread_decoder_decoder_block_1_conv2d_7_kernel:ђ@L
>read_9_disablecopyonread_decoder_decoder_block_1_conv2d_7_bias:@[
Mread_10_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_gamma:@Z
Lread_11_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_beta:@a
Sread_12_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_moving_mean:@e
Wread_13_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_moving_variance:@[
Aread_14_disablecopyonread_decoder_decoder_block_2_conv2d_8_kernel:@ M
?read_15_disablecopyonread_decoder_decoder_block_2_conv2d_8_bias: [
Mread_16_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_gamma: Z
Lread_17_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_beta: a
Sread_18_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_moving_mean: e
Wread_19_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_moving_variance: K
1read_20_disablecopyonread_decoder_conv2d_9_kernel: =
/read_21_disablecopyonread_decoder_conv2d_9_bias:L
2read_22_disablecopyonread_decoder_conv2d_10_kernel:>
0read_23_disablecopyonread_decoder_conv2d_10_bias:
savev2_const
identity_49ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_19/DisableCopyOnReadбRead_19/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_20/DisableCopyOnReadбRead_20/ReadVariableOpбRead_21/DisableCopyOnReadбRead_21/ReadVariableOpбRead_22/DisableCopyOnReadбRead_22/ReadVariableOpбRead_23/DisableCopyOnReadбRead_23/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead-read_disablecopyonread_decoder_dense_2_kernel"/device:CPU:0*
_output_shapes
 г
Read/ReadVariableOpReadVariableOp-read_disablecopyonread_decoder_dense_2_kernel^Read/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:ђђђ*
dtype0l
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:ђђђd

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*!
_output_shapes
:ђђђЂ
Read_1/DisableCopyOnReadDisableCopyOnRead-read_1_disablecopyonread_decoder_dense_2_bias"/device:CPU:0*
_output_shapes
 Ф
Read_1/ReadVariableOpReadVariableOp-read_1_disablecopyonread_decoder_dense_2_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:ђђ*
dtype0k

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:ђђa

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:ђђњ
Read_2/DisableCopyOnReadDisableCopyOnRead>read_2_disablecopyonread_decoder_decoder_block_conv2d_6_kernel"/device:CPU:0*
_output_shapes
 ╚
Read_2/ReadVariableOpReadVariableOp>read_2_disablecopyonread_decoder_decoder_block_conv2d_6_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:ђђ*
dtype0w

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ђђm

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*(
_output_shapes
:ђђљ
Read_3/DisableCopyOnReadDisableCopyOnRead<read_3_disablecopyonread_decoder_decoder_block_conv2d_6_bias"/device:CPU:0*
_output_shapes
 ╣
Read_3/ReadVariableOpReadVariableOp<read_3_disablecopyonread_decoder_decoder_block_conv2d_6_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђъ
Read_4/DisableCopyOnReadDisableCopyOnReadJread_4_disablecopyonread_decoder_decoder_block_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 К
Read_4/ReadVariableOpReadVariableOpJread_4_disablecopyonread_decoder_decoder_block_batch_normalization_5_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЮ
Read_5/DisableCopyOnReadDisableCopyOnReadIread_5_disablecopyonread_decoder_decoder_block_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 к
Read_5/ReadVariableOpReadVariableOpIread_5_disablecopyonread_decoder_decoder_block_batch_normalization_5_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђц
Read_6/DisableCopyOnReadDisableCopyOnReadPread_6_disablecopyonread_decoder_decoder_block_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 ═
Read_6/ReadVariableOpReadVariableOpPread_6_disablecopyonread_decoder_decoder_block_batch_normalization_5_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђе
Read_7/DisableCopyOnReadDisableCopyOnReadTread_7_disablecopyonread_decoder_decoder_block_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 Л
Read_7/ReadVariableOpReadVariableOpTread_7_disablecopyonread_decoder_decoder_block_batch_normalization_5_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђћ
Read_8/DisableCopyOnReadDisableCopyOnRead@read_8_disablecopyonread_decoder_decoder_block_1_conv2d_7_kernel"/device:CPU:0*
_output_shapes
 ╔
Read_8/ReadVariableOpReadVariableOp@read_8_disablecopyonread_decoder_decoder_block_1_conv2d_7_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:ђ@*
dtype0w
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:ђ@n
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*'
_output_shapes
:ђ@њ
Read_9/DisableCopyOnReadDisableCopyOnRead>read_9_disablecopyonread_decoder_decoder_block_1_conv2d_7_bias"/device:CPU:0*
_output_shapes
 ║
Read_9/ReadVariableOpReadVariableOp>read_9_disablecopyonread_decoder_decoder_block_1_conv2d_7_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
:@б
Read_10/DisableCopyOnReadDisableCopyOnReadMread_10_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_10/ReadVariableOpReadVariableOpMread_10_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
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
:@А
Read_11/DisableCopyOnReadDisableCopyOnReadLread_11_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 ╩
Read_11/ReadVariableOpReadVariableOpLread_11_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
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
:@е
Read_12/DisableCopyOnReadDisableCopyOnReadSread_12_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_moving_mean"/device:CPU:0*
_output_shapes
 Л
Read_12/ReadVariableOpReadVariableOpSread_12_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
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
:@г
Read_13/DisableCopyOnReadDisableCopyOnReadWread_13_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_13/ReadVariableOpReadVariableOpWread_13_disablecopyonread_decoder_decoder_block_1_batch_normalization_6_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
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
:@ќ
Read_14/DisableCopyOnReadDisableCopyOnReadAread_14_disablecopyonread_decoder_decoder_block_2_conv2d_8_kernel"/device:CPU:0*
_output_shapes
 ╦
Read_14/ReadVariableOpReadVariableOpAread_14_disablecopyonread_decoder_decoder_block_2_conv2d_8_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
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
:@ ћ
Read_15/DisableCopyOnReadDisableCopyOnRead?read_15_disablecopyonread_decoder_decoder_block_2_conv2d_8_bias"/device:CPU:0*
_output_shapes
 й
Read_15/ReadVariableOpReadVariableOp?read_15_disablecopyonread_decoder_decoder_block_2_conv2d_8_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
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
: б
Read_16/DisableCopyOnReadDisableCopyOnReadMread_16_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_16/ReadVariableOpReadVariableOpMread_16_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
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
: А
Read_17/DisableCopyOnReadDisableCopyOnReadLread_17_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 ╩
Read_17/ReadVariableOpReadVariableOpLread_17_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
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
: е
Read_18/DisableCopyOnReadDisableCopyOnReadSread_18_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_moving_mean"/device:CPU:0*
_output_shapes
 Л
Read_18/ReadVariableOpReadVariableOpSread_18_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
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
: г
Read_19/DisableCopyOnReadDisableCopyOnReadWread_19_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_19/ReadVariableOpReadVariableOpWread_19_disablecopyonread_decoder_decoder_block_2_batch_normalization_7_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
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
: є
Read_20/DisableCopyOnReadDisableCopyOnRead1read_20_disablecopyonread_decoder_conv2d_9_kernel"/device:CPU:0*
_output_shapes
 ╗
Read_20/ReadVariableOpReadVariableOp1read_20_disablecopyonread_decoder_conv2d_9_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
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
: ё
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_decoder_conv2d_9_bias"/device:CPU:0*
_output_shapes
 Г
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_decoder_conv2d_9_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
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
:Є
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_decoder_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_decoder_conv2d_10_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
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
:Ё
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_decoder_conv2d_10_bias"/device:CPU:0*
_output_shapes
 «
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_decoder_conv2d_10_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
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
:п
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ђ
valueэBЗB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЪ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B ч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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
: ╗

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
┐
M
1__inference_up_sampling2d_1_layer_call_fn_1905790

inputs
identity▀
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1904050Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ї
Ј
%__inference_signature_wrapper_1905060
input_1
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ$
	unknown_7:ђ@
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
identityѕбStatefulPartitionedCallЫ
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
:         ђђ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *+
f&R$
"__inference__wrapped_model_1903954y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
Еб
э
"__inference__wrapped_model_1903954
input_1C
.decoder_dense_2_matmul_readvariableop_resource:ђђђ?
/decoder_dense_2_biasadd_readvariableop_resource:
ђђY
=decoder_decoder_block_conv2d_6_conv2d_readvariableop_resource:ђђM
>decoder_decoder_block_conv2d_6_biasadd_readvariableop_resource:	ђR
Cdecoder_decoder_block_batch_normalization_5_readvariableop_resource:	ђT
Edecoder_decoder_block_batch_normalization_5_readvariableop_1_resource:	ђc
Tdecoder_decoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђe
Vdecoder_decoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђZ
?decoder_decoder_block_1_conv2d_7_conv2d_readvariableop_resource:ђ@N
@decoder_decoder_block_1_conv2d_7_biasadd_readvariableop_resource:@S
Edecoder_decoder_block_1_batch_normalization_6_readvariableop_resource:@U
Gdecoder_decoder_block_1_batch_normalization_6_readvariableop_1_resource:@d
Vdecoder_decoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@f
Xdecoder_decoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@Y
?decoder_decoder_block_2_conv2d_8_conv2d_readvariableop_resource:@ N
@decoder_decoder_block_2_conv2d_8_biasadd_readvariableop_resource: S
Edecoder_decoder_block_2_batch_normalization_7_readvariableop_resource: U
Gdecoder_decoder_block_2_batch_normalization_7_readvariableop_1_resource: d
Vdecoder_decoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: f
Xdecoder_decoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: I
/decoder_conv2d_9_conv2d_readvariableop_resource: >
0decoder_conv2d_9_biasadd_readvariableop_resource:J
0decoder_conv2d_10_conv2d_readvariableop_resource:?
1decoder_conv2d_10_biasadd_readvariableop_resource:
identityѕб(decoder/conv2d_10/BiasAdd/ReadVariableOpб'decoder/conv2d_10/Conv2D/ReadVariableOpб'decoder/conv2d_9/BiasAdd/ReadVariableOpб&decoder/conv2d_9/Conv2D/ReadVariableOpбKdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбMdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б:decoder/decoder_block/batch_normalization_5/ReadVariableOpб<decoder/decoder_block/batch_normalization_5/ReadVariableOp_1б5decoder/decoder_block/conv2d_6/BiasAdd/ReadVariableOpб4decoder/decoder_block/conv2d_6/Conv2D/ReadVariableOpбMdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpбOdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б<decoder/decoder_block_1/batch_normalization_6/ReadVariableOpб>decoder/decoder_block_1/batch_normalization_6/ReadVariableOp_1б7decoder/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOpб6decoder/decoder_block_1/conv2d_7/Conv2D/ReadVariableOpбMdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбOdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б<decoder/decoder_block_2/batch_normalization_7/ReadVariableOpб>decoder/decoder_block_2/batch_normalization_7/ReadVariableOp_1б7decoder/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOpб6decoder/decoder_block_2/conv2d_8/Conv2D/ReadVariableOpб&decoder/dense_2/BiasAdd/ReadVariableOpб%decoder/dense_2/MatMul/ReadVariableOpЌ
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0ї
decoder/dense_2/MatMulMatMulinput_1-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђћ
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0е
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђr
decoder/dense_2/ReluRelu decoder/dense_2/BiasAdd:output:0*
T0*)
_output_shapes
:         ђђn
decoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ў
decoder/ReshapeReshape"decoder/dense_2/Relu:activations:0decoder/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђ╝
4decoder/decoder_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp=decoder_decoder_block_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0Ж
%decoder/decoder_block/conv2d_6/Conv2DConv2Ddecoder/Reshape:output:0<decoder/decoder_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
▒
5decoder/decoder_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp>decoder_decoder_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0█
&decoder/decoder_block/conv2d_6/BiasAddBiasAdd.decoder/decoder_block/conv2d_6/Conv2D:output:0=decoder/decoder_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЌ
#decoder/decoder_block/conv2d_6/ReluRelu/decoder/decoder_block/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђz
)decoder/decoder_block/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      |
+decoder/decoder_block/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      й
'decoder/decoder_block/up_sampling2d/mulMul2decoder/decoder_block/up_sampling2d/Const:output:04decoder/decoder_block/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:ј
@decoder/decoder_block/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor1decoder/decoder_block/conv2d_6/Relu:activations:0+decoder/decoder_block/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(╗
:decoder/decoder_block/batch_normalization_5/ReadVariableOpReadVariableOpCdecoder_decoder_block_batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┐
<decoder/decoder_block/batch_normalization_5/ReadVariableOp_1ReadVariableOpEdecoder_decoder_block_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0П
Kdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpTdecoder_decoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0р
Mdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVdecoder_decoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Р
<decoder/decoder_block/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3Qdecoder/decoder_block/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Bdecoder/decoder_block/batch_normalization_5/ReadVariableOp:value:0Ddecoder/decoder_block/batch_normalization_5/ReadVariableOp_1:value:0Sdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Udecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ┐
6decoder/decoder_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp?decoder_decoder_block_1_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0Ћ
'decoder/decoder_block_1/conv2d_7/Conv2DConv2D@decoder/decoder_block/batch_normalization_5/FusedBatchNormV3:y:0>decoder/decoder_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
┤
7decoder/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp@decoder_decoder_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Я
(decoder/decoder_block_1/conv2d_7/BiasAddBiasAdd0decoder/decoder_block_1/conv2d_7/Conv2D:output:0?decoder/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @џ
%decoder/decoder_block_1/conv2d_7/ReluRelu1decoder/decoder_block_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @~
-decoder/decoder_block_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      ђ
/decoder/decoder_block_1/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ╔
+decoder/decoder_block_1/up_sampling2d_1/mulMul6decoder/decoder_block_1/up_sampling2d_1/Const:output:08decoder/decoder_block_1/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:Ќ
Ddecoder/decoder_block_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor3decoder/decoder_block_1/conv2d_7/Relu:activations:0/decoder/decoder_block_1/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(Й
<decoder/decoder_block_1/batch_normalization_6/ReadVariableOpReadVariableOpEdecoder_decoder_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0┬
>decoder/decoder_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpGdecoder_decoder_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0Я
Mdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpVdecoder_decoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0С
Odecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXdecoder_decoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0в
>decoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3Udecoder/decoder_block_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0Ddecoder/decoder_block_1/batch_normalization_6/ReadVariableOp:value:0Fdecoder/decoder_block_1/batch_normalization_6/ReadVariableOp_1:value:0Udecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Wdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
is_training( Й
6decoder/decoder_block_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp?decoder_decoder_block_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ќ
'decoder/decoder_block_2/conv2d_8/Conv2DConv2DBdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3:y:0>decoder/decoder_block_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
┤
7decoder/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp@decoder_decoder_block_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Я
(decoder/decoder_block_2/conv2d_8/BiasAddBiasAdd0decoder/decoder_block_2/conv2d_8/Conv2D:output:0?decoder/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            џ
%decoder/decoder_block_2/conv2d_8/ReluRelu1decoder/decoder_block_2/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            ~
-decoder/decoder_block_2/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        ђ
/decoder/decoder_block_2/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ╔
+decoder/decoder_block_2/up_sampling2d_2/mulMul6decoder/decoder_block_2/up_sampling2d_2/Const:output:08decoder/decoder_block_2/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Ў
Ddecoder/decoder_block_2/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor3decoder/decoder_block_2/conv2d_8/Relu:activations:0/decoder/decoder_block_2/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(Й
<decoder/decoder_block_2/batch_normalization_7/ReadVariableOpReadVariableOpEdecoder_decoder_block_2_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0┬
>decoder/decoder_block_2/batch_normalization_7/ReadVariableOp_1ReadVariableOpGdecoder_decoder_block_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0Я
Mdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpVdecoder_decoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0С
Odecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpXdecoder_decoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ь
>decoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3Udecoder/decoder_block_2/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0Ddecoder/decoder_block_2/batch_normalization_7/ReadVariableOp:value:0Fdecoder/decoder_block_2/batch_normalization_7/ReadVariableOp_1:value:0Udecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Wdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
is_training( ъ
&decoder/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0щ
decoder/conv2d_9/Conv2DConv2DBdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3:y:0.decoder/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
ћ
'decoder/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▓
decoder/conv2d_9/BiasAddBiasAdd decoder/conv2d_9/Conv2D:output:0/decoder/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ|
decoder/conv2d_9/ReluRelu!decoder/conv2d_9/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђа
'decoder/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0▄
decoder/conv2d_10/Conv2DConv2D#decoder/conv2d_9/Relu:activations:0/decoder/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
ќ
(decoder/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
decoder/conv2d_10/BiasAddBiasAdd!decoder/conv2d_10/Conv2D:output:00decoder/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђё
decoder/conv2d_10/SigmoidSigmoid"decoder/conv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђv
IdentityIdentitydecoder/conv2d_10/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђЫ
NoOpNoOp)^decoder/conv2d_10/BiasAdd/ReadVariableOp(^decoder/conv2d_10/Conv2D/ReadVariableOp(^decoder/conv2d_9/BiasAdd/ReadVariableOp'^decoder/conv2d_9/Conv2D/ReadVariableOpL^decoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpN^decoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1;^decoder/decoder_block/batch_normalization_5/ReadVariableOp=^decoder/decoder_block/batch_normalization_5/ReadVariableOp_16^decoder/decoder_block/conv2d_6/BiasAdd/ReadVariableOp5^decoder/decoder_block/conv2d_6/Conv2D/ReadVariableOpN^decoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpP^decoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1=^decoder/decoder_block_1/batch_normalization_6/ReadVariableOp?^decoder/decoder_block_1/batch_normalization_6/ReadVariableOp_18^decoder/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp7^decoder/decoder_block_1/conv2d_7/Conv2D/ReadVariableOpN^decoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpP^decoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1=^decoder/decoder_block_2/batch_normalization_7/ReadVariableOp?^decoder/decoder_block_2/batch_normalization_7/ReadVariableOp_18^decoder/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp7^decoder/decoder_block_2/conv2d_8/Conv2D/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2T
(decoder/conv2d_10/BiasAdd/ReadVariableOp(decoder/conv2d_10/BiasAdd/ReadVariableOp2R
'decoder/conv2d_10/Conv2D/ReadVariableOp'decoder/conv2d_10/Conv2D/ReadVariableOp2R
'decoder/conv2d_9/BiasAdd/ReadVariableOp'decoder/conv2d_9/BiasAdd/ReadVariableOp2P
&decoder/conv2d_9/Conv2D/ReadVariableOp&decoder/conv2d_9/Conv2D/ReadVariableOp2ъ
Mdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Mdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12џ
Kdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpKdecoder/decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2|
<decoder/decoder_block/batch_normalization_5/ReadVariableOp_1<decoder/decoder_block/batch_normalization_5/ReadVariableOp_12x
:decoder/decoder_block/batch_normalization_5/ReadVariableOp:decoder/decoder_block/batch_normalization_5/ReadVariableOp2n
5decoder/decoder_block/conv2d_6/BiasAdd/ReadVariableOp5decoder/decoder_block/conv2d_6/BiasAdd/ReadVariableOp2l
4decoder/decoder_block/conv2d_6/Conv2D/ReadVariableOp4decoder/decoder_block/conv2d_6/Conv2D/ReadVariableOp2б
Odecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Odecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12ъ
Mdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpMdecoder/decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2ђ
>decoder/decoder_block_1/batch_normalization_6/ReadVariableOp_1>decoder/decoder_block_1/batch_normalization_6/ReadVariableOp_12|
<decoder/decoder_block_1/batch_normalization_6/ReadVariableOp<decoder/decoder_block_1/batch_normalization_6/ReadVariableOp2r
7decoder/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp7decoder/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp2p
6decoder/decoder_block_1/conv2d_7/Conv2D/ReadVariableOp6decoder/decoder_block_1/conv2d_7/Conv2D/ReadVariableOp2б
Odecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Odecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12ъ
Mdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpMdecoder/decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2ђ
>decoder/decoder_block_2/batch_normalization_7/ReadVariableOp_1>decoder/decoder_block_2/batch_normalization_7/ReadVariableOp_12|
<decoder/decoder_block_2/batch_normalization_7/ReadVariableOp<decoder/decoder_block_2/batch_normalization_7/ReadVariableOp2r
7decoder/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp7decoder/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp2p
6decoder/decoder_block_2/conv2d_8/Conv2D/ReadVariableOp6decoder/decoder_block_2/conv2d_8/Conv2D/ReadVariableOp2P
&decoder/dense_2/BiasAdd/ReadVariableOp&decoder/dense_2/BiasAdd/ReadVariableOp2N
%decoder/dense_2/MatMul/ReadVariableOp%decoder/dense_2/MatMul/ReadVariableOp:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
»

Щ
D__inference_dense_2_layer_call_and_return_conditional_losses_1904218

inputs3
matmul_readvariableop_resource:ђђђ/
biasadd_readvariableop_resource:
ђђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0k
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђt
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0x
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђR
ReluReluBiasAdd:output:0*
T0*)
_output_shapes
:         ђђc
IdentityIdentityRelu:activations:0^NoOp*
T0*)
_output_shapes
:         ђђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ќ
┼
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905767

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
П
А
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1904010

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
љ
■
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1904366

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ђђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ 
 
_user_specified_nameinputs
И	
Ћ
/__inference_decoder_block_layer_call_fn_1905424
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904428x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
Й"
љ
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904428
input_tensorC
'conv2d_6_conv2d_readvariableop_resource:ђђ7
(conv2d_6_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1бconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpљ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▓
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ё
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ў
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђk
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђd
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
:╠
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_6/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Ј
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0я
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ѓ
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђ╦
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
«њ
А
D__inference_decoder_layer_call_and_return_conditional_losses_1905370
embedding_input;
&dense_2_matmul_readvariableop_resource:ђђђ7
'dense_2_biasadd_readvariableop_resource:
ђђQ
5decoder_block_conv2d_6_conv2d_readvariableop_resource:ђђE
6decoder_block_conv2d_6_biasadd_readvariableop_resource:	ђJ
;decoder_block_batch_normalization_5_readvariableop_resource:	ђL
=decoder_block_batch_normalization_5_readvariableop_1_resource:	ђ[
Ldecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђ]
Ndecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђR
7decoder_block_1_conv2d_7_conv2d_readvariableop_resource:ђ@F
8decoder_block_1_conv2d_7_biasadd_readvariableop_resource:@K
=decoder_block_1_batch_normalization_6_readvariableop_resource:@M
?decoder_block_1_batch_normalization_6_readvariableop_1_resource:@\
Ndecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@^
Pdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@Q
7decoder_block_2_conv2d_8_conv2d_readvariableop_resource:@ F
8decoder_block_2_conv2d_8_biasadd_readvariableop_resource: K
=decoder_block_2_batch_normalization_7_readvariableop_resource: M
?decoder_block_2_batch_normalization_7_readvariableop_1_resource: \
Ndecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource: ^
Pdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:
identityѕб conv2d_10/BiasAdd/ReadVariableOpбconv2d_10/Conv2D/ReadVariableOpбconv2d_9/BiasAdd/ReadVariableOpбconv2d_9/Conv2D/ReadVariableOpбCdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбEdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б2decoder_block/batch_normalization_5/ReadVariableOpб4decoder_block/batch_normalization_5/ReadVariableOp_1б-decoder_block/conv2d_6/BiasAdd/ReadVariableOpб,decoder_block/conv2d_6/Conv2D/ReadVariableOpбEdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpбGdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б4decoder_block_1/batch_normalization_6/ReadVariableOpб6decoder_block_1/batch_normalization_6/ReadVariableOp_1б/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOpб.decoder_block_1/conv2d_7/Conv2D/ReadVariableOpбEdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбGdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б4decoder_block_2/batch_normalization_7/ReadVariableOpб6decoder_block_2/batch_normalization_7/ReadVariableOp_1б/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOpб.decoder_block_2/conv2d_8/Conv2D/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpбdense_2/MatMul/ReadVariableOpЄ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0ё
dense_2/MatMulMatMulembedding_input%dense_2/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђё
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0љ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђb
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*)
_output_shapes
:         ђђf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ђ
ReshapeReshapedense_2/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђг
,decoder_block/conv2d_6/Conv2D/ReadVariableOpReadVariableOp5decoder_block_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0м
decoder_block/conv2d_6/Conv2DConv2DReshape:output:04decoder_block/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
А
-decoder_block/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6decoder_block_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0├
decoder_block/conv2d_6/BiasAddBiasAdd&decoder_block/conv2d_6/Conv2D:output:05decoder_block/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЄ
decoder_block/conv2d_6/ReluRelu'decoder_block/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђr
!decoder_block/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      t
#decoder_block/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ц
decoder_block/up_sampling2d/mulMul*decoder_block/up_sampling2d/Const:output:0,decoder_block/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:Ш
8decoder_block/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor)decoder_block/conv2d_6/Relu:activations:0#decoder_block/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Ф
2decoder_block/batch_normalization_5/ReadVariableOpReadVariableOp;decoder_block_batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0»
4decoder_block/batch_normalization_5/ReadVariableOp_1ReadVariableOp=decoder_block_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
Cdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpLdecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Л
Edecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNdecoder_block_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▓
4decoder_block/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3Idecoder_block/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:decoder_block/batch_normalization_5/ReadVariableOp:value:0<decoder_block/batch_normalization_5/ReadVariableOp_1:value:0Kdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Mdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( »
.decoder_block_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp7decoder_block_1_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0§
decoder_block_1/conv2d_7/Conv2DConv2D8decoder_block/batch_normalization_5/FusedBatchNormV3:y:06decoder_block_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ц
/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╚
 decoder_block_1/conv2d_7/BiasAddBiasAdd(decoder_block_1/conv2d_7/Conv2D:output:07decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @і
decoder_block_1/conv2d_7/ReluRelu)decoder_block_1/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @v
%decoder_block_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      x
'decoder_block_1/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ▒
#decoder_block_1/up_sampling2d_1/mulMul.decoder_block_1/up_sampling2d_1/Const:output:00decoder_block_1/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
: 
<decoder_block_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor+decoder_block_1/conv2d_7/Relu:activations:0'decoder_block_1/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(«
4decoder_block_1/batch_normalization_6/ReadVariableOpReadVariableOp=decoder_block_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0▓
6decoder_block_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp?decoder_block_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0л
Edecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╗
6decoder_block_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0<decoder_block_1/batch_normalization_6/ReadVariableOp:value:0>decoder_block_1/batch_normalization_6/ReadVariableOp_1:value:0Mdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
is_training( «
.decoder_block_2/conv2d_8/Conv2D/ReadVariableOpReadVariableOp7decoder_block_2_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0 
decoder_block_2/conv2d_8/Conv2DConv2D:decoder_block_1/batch_normalization_6/FusedBatchNormV3:y:06decoder_block_2/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
ц
/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp8decoder_block_2_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╚
 decoder_block_2/conv2d_8/BiasAddBiasAdd(decoder_block_2/conv2d_8/Conv2D:output:07decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            і
decoder_block_2/conv2d_8/ReluRelu)decoder_block_2/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            v
%decoder_block_2/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        x
'decoder_block_2/up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ▒
#decoder_block_2/up_sampling2d_2/mulMul.decoder_block_2/up_sampling2d_2/Const:output:00decoder_block_2/up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Ђ
<decoder_block_2/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor+decoder_block_2/conv2d_8/Relu:activations:0'decoder_block_2/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(«
4decoder_block_2/batch_normalization_7/ReadVariableOpReadVariableOp=decoder_block_2_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0▓
6decoder_block_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp?decoder_block_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0л
Edecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpNdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0н
Gdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPdecoder_block_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0й
6decoder_block_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3Mdecoder_block_2/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0<decoder_block_2/batch_normalization_7/ReadVariableOp:value:0>decoder_block_2/batch_normalization_7/ReadVariableOp_1:value:0Mdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Odecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
is_training( ј
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0р
conv2d_9/Conv2DConv2D:decoder_block_2/batch_normalization_7/FusedBatchNormV3:y:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
ё
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0џ
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђl
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђљ
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0─
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
є
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђt
conv2d_10/SigmoidSigmoidconv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђn
IdentityIdentityconv2d_10/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђ▓

NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOpD^decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpF^decoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_13^decoder_block/batch_normalization_5/ReadVariableOp5^decoder_block/batch_normalization_5/ReadVariableOp_1.^decoder_block/conv2d_6/BiasAdd/ReadVariableOp-^decoder_block/conv2d_6/Conv2D/ReadVariableOpF^decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpH^decoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_15^decoder_block_1/batch_normalization_6/ReadVariableOp7^decoder_block_1/batch_normalization_6/ReadVariableOp_10^decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp/^decoder_block_1/conv2d_7/Conv2D/ReadVariableOpF^decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpH^decoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_15^decoder_block_2/batch_normalization_7/ReadVariableOp7^decoder_block_2/batch_normalization_7/ReadVariableOp_10^decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp/^decoder_block_2/conv2d_8/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2ј
Edecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Edecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12і
Cdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOpCdecoder_block/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2l
4decoder_block/batch_normalization_5/ReadVariableOp_14decoder_block/batch_normalization_5/ReadVariableOp_12h
2decoder_block/batch_normalization_5/ReadVariableOp2decoder_block/batch_normalization_5/ReadVariableOp2^
-decoder_block/conv2d_6/BiasAdd/ReadVariableOp-decoder_block/conv2d_6/BiasAdd/ReadVariableOp2\
,decoder_block/conv2d_6/Conv2D/ReadVariableOp,decoder_block/conv2d_6/Conv2D/ReadVariableOp2њ
Gdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12ј
Edecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpEdecoder_block_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_1/batch_normalization_6/ReadVariableOp_16decoder_block_1/batch_normalization_6/ReadVariableOp_12l
4decoder_block_1/batch_normalization_6/ReadVariableOp4decoder_block_1/batch_normalization_6/ReadVariableOp2b
/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp/decoder_block_1/conv2d_7/BiasAdd/ReadVariableOp2`
.decoder_block_1/conv2d_7/Conv2D/ReadVariableOp.decoder_block_1/conv2d_7/Conv2D/ReadVariableOp2њ
Gdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Gdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12ј
Edecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpEdecoder_block_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2p
6decoder_block_2/batch_normalization_7/ReadVariableOp_16decoder_block_2/batch_normalization_7/ReadVariableOp_12l
4decoder_block_2/batch_normalization_7/ReadVariableOp4decoder_block_2/batch_normalization_7/ReadVariableOp2b
/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp/decoder_block_2/conv2d_8/BiasAdd/ReadVariableOp2`
.decoder_block_2/conv2d_8/Conv2D/ReadVariableOp.decoder_block_2/conv2d_8/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
╦
Џ
)__inference_decoder_layer_call_fn_1905166
embedding_input
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ$
	unknown_7:ђ@
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
identityѕбStatefulPartitionedCallю
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
:         ђђ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1904713y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
О)
█
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905637
input_tensorA
'conv2d_8_conv2d_readvariableop_resource:@ 6
(conv2d_8_biasadd_readvariableop_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб$batch_normalization_7/AssignNewValueб&batch_normalization_7/AssignNewValue_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1бconv2d_8/BiasAdd/ReadVariableOpбconv2d_8/Conv2D/ReadVariableOpј
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0▒
conv2d_8/Conv2DConv2Dinput_tensor&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
ё
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            f
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Л
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_8/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(ј
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0в
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ Џ
NoOpNoOp%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':           @: : : : : : 2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:           @
&
_user_specified_nameinput_tensor
│
Њ
)__inference_decoder_layer_call_fn_1904764
input_1
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ$
	unknown_7:ђ@
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
identityѕбStatefulPartitionedCallћ
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
:         ђђ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1904713y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
Н+
љ

D__inference_decoder_layer_call_and_return_conditional_losses_1904713
embedding_input$
dense_2_1904656:ђђђ
dense_2_1904658:
ђђ1
decoder_block_1904663:ђђ$
decoder_block_1904665:	ђ$
decoder_block_1904667:	ђ$
decoder_block_1904669:	ђ$
decoder_block_1904671:	ђ$
decoder_block_1904673:	ђ2
decoder_block_1_1904676:ђ@%
decoder_block_1_1904678:@%
decoder_block_1_1904680:@%
decoder_block_1_1904682:@%
decoder_block_1_1904684:@%
decoder_block_1_1904686:@1
decoder_block_2_1904689:@ %
decoder_block_2_1904691: %
decoder_block_2_1904693: %
decoder_block_2_1904695: %
decoder_block_2_1904697: %
decoder_block_2_1904699: *
conv2d_9_1904702: 
conv2d_9_1904704:+
conv2d_10_1904707:
conv2d_10_1904709:
identityѕб!conv2d_10/StatefulPartitionedCallб conv2d_9/StatefulPartitionedCallб%decoder_block/StatefulPartitionedCallб'decoder_block_1/StatefulPartitionedCallб'decoder_block_2/StatefulPartitionedCallбdense_2/StatefulPartitionedCall 
dense_2/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_2_1904656dense_2_1904658*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1904218f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ј
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђЃ
%decoder_block/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_1904663decoder_block_1904665decoder_block_1904667decoder_block_1904669decoder_block_1904671decoder_block_1904673*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904428░
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_block/StatefulPartitionedCall:output:0decoder_block_1_1904676decoder_block_1_1904678decoder_block_1_1904680decoder_block_1_1904682decoder_block_1_1904684decoder_block_1_1904686*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904470┤
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0decoder_block_2_1904689decoder_block_2_1904691decoder_block_2_1904693decoder_block_2_1904695decoder_block_2_1904697decoder_block_2_1904699*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904512г
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_9_1904702conv2d_9_1904704*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1904366Е
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_1904707conv2d_10_1904709*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1904383Ѓ
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђФ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
Н)
▄
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904298
input_tensorB
'conv2d_7_conv2d_readvariableop_resource:ђ@6
(conv2d_7_biasadd_readvariableop_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_6/AssignNewValueб&batch_normalization_6/AssignNewValue_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1бconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpЈ
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0▒
conv2d_7/Conv2DConv2Dinput_tensor&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:¤
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_7/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(ј
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ђ
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:           @Џ
NoOpNoOp%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
п)
Я
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904255
input_tensorC
'conv2d_6_conv2d_readvariableop_resource:ђђ7
(conv2d_6_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1бconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpљ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▓
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ё
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ў
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђk
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђd
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
:╠
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_6/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Ј
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0В
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ѓ
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђЏ
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
м
Џ
)__inference_dense_2_layer_call_fn_1905379

inputs
unknown:ђђђ
	unknown_0:
ђђ
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1904218q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ќ	
м
7__inference_batch_normalization_6_layer_call_fn_1905815

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1904075Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ќ
┼
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1903992

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
а	
о
7__inference_batch_normalization_5_layer_call_fn_1905749

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1904010і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
жn
▄
#__inference__traced_restore_1906192
file_prefix<
'assignvariableop_decoder_dense_2_kernel:ђђђ7
'assignvariableop_1_decoder_dense_2_bias:
ђђT
8assignvariableop_2_decoder_decoder_block_conv2d_6_kernel:ђђE
6assignvariableop_3_decoder_decoder_block_conv2d_6_bias:	ђS
Dassignvariableop_4_decoder_decoder_block_batch_normalization_5_gamma:	ђR
Cassignvariableop_5_decoder_decoder_block_batch_normalization_5_beta:	ђY
Jassignvariableop_6_decoder_decoder_block_batch_normalization_5_moving_mean:	ђ]
Nassignvariableop_7_decoder_decoder_block_batch_normalization_5_moving_variance:	ђU
:assignvariableop_8_decoder_decoder_block_1_conv2d_7_kernel:ђ@F
8assignvariableop_9_decoder_decoder_block_1_conv2d_7_bias:@U
Gassignvariableop_10_decoder_decoder_block_1_batch_normalization_6_gamma:@T
Fassignvariableop_11_decoder_decoder_block_1_batch_normalization_6_beta:@[
Massignvariableop_12_decoder_decoder_block_1_batch_normalization_6_moving_mean:@_
Qassignvariableop_13_decoder_decoder_block_1_batch_normalization_6_moving_variance:@U
;assignvariableop_14_decoder_decoder_block_2_conv2d_8_kernel:@ G
9assignvariableop_15_decoder_decoder_block_2_conv2d_8_bias: U
Gassignvariableop_16_decoder_decoder_block_2_batch_normalization_7_gamma: T
Fassignvariableop_17_decoder_decoder_block_2_batch_normalization_7_beta: [
Massignvariableop_18_decoder_decoder_block_2_batch_normalization_7_moving_mean: _
Qassignvariableop_19_decoder_decoder_block_2_batch_normalization_7_moving_variance: E
+assignvariableop_20_decoder_conv2d_9_kernel: 7
)assignvariableop_21_decoder_conv2d_9_bias:F
,assignvariableop_22_decoder_conv2d_10_kernel:8
*assignvariableop_23_decoder_conv2d_10_bias:
identity_25ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9█
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ђ
valueэBЗB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHб
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Џ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOpAssignVariableOp'assignvariableop_decoder_dense_2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_1AssignVariableOp'assignvariableop_1_decoder_dense_2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_2AssignVariableOp8assignvariableop_2_decoder_decoder_block_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOp_3AssignVariableOp6assignvariableop_3_decoder_decoder_block_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:█
AssignVariableOp_4AssignVariableOpDassignvariableop_4_decoder_decoder_block_batch_normalization_5_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:┌
AssignVariableOp_5AssignVariableOpCassignvariableop_5_decoder_decoder_block_batch_normalization_5_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_6AssignVariableOpJassignvariableop_6_decoder_decoder_block_batch_normalization_5_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_7AssignVariableOpNassignvariableop_7_decoder_decoder_block_batch_normalization_5_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_8AssignVariableOp:assignvariableop_8_decoder_decoder_block_1_conv2d_7_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_9AssignVariableOp8assignvariableop_9_decoder_decoder_block_1_conv2d_7_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_10AssignVariableOpGassignvariableop_10_decoder_decoder_block_1_batch_normalization_6_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_11AssignVariableOpFassignvariableop_11_decoder_decoder_block_1_batch_normalization_6_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_12AssignVariableOpMassignvariableop_12_decoder_decoder_block_1_batch_normalization_6_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_13AssignVariableOpQassignvariableop_13_decoder_decoder_block_1_batch_normalization_6_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_14AssignVariableOp;assignvariableop_14_decoder_decoder_block_2_conv2d_8_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_15AssignVariableOp9assignvariableop_15_decoder_decoder_block_2_conv2d_8_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_16AssignVariableOpGassignvariableop_16_decoder_decoder_block_2_batch_normalization_7_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_17AssignVariableOpFassignvariableop_17_decoder_decoder_block_2_batch_normalization_7_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_18AssignVariableOpMassignvariableop_18_decoder_decoder_block_2_batch_normalization_7_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_19AssignVariableOpQassignvariableop_19_decoder_decoder_block_2_batch_normalization_7_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_20AssignVariableOp+assignvariableop_20_decoder_conv2d_9_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_21AssignVariableOp)assignvariableop_21_decoder_conv2d_9_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_22AssignVariableOp,assignvariableop_22_decoder_conv2d_10_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_23AssignVariableOp*assignvariableop_23_decoder_conv2d_10_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ▀
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ╠
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
┼
Џ
)__inference_decoder_layer_call_fn_1905113
embedding_input
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ$
	unknown_7:ђ@
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
identityѕбStatefulPartitionedCallќ
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
:         ђђ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1904600y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
П
А
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905785

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
љ
 
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1905706

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
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
:         ђђ`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:         ђђd
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
═
Ю
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1904093

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ъ	
о
7__inference_batch_normalization_5_layer_call_fn_1905736

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1903992і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1904075

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╗"
ї
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904470
input_tensorB
'conv2d_7_conv2d_readvariableop_resource:ђ@6
(conv2d_7_biasadd_readvariableop_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1бconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpЈ
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0▒
conv2d_7/Conv2DConv2Dinput_tensor&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:¤
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_7/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(ј
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0█
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
is_training( Ђ
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:           @╦
NoOpNoOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
Є
┴
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905846

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
х	
љ
1__inference_decoder_block_2_layer_call_fn_1905608
input_tensor!
unknown:@ 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904512y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':           @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:           @
&
_user_specified_nameinput_tensor
љ
 
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1904383

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
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
:         ђђ`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:         ђђd
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ў	
м
7__inference_batch_normalization_6_layer_call_fn_1905828

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1904093Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╗"
ї
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905574
input_tensorB
'conv2d_7_conv2d_readvariableop_resource:ђ@6
(conv2d_7_biasadd_readvariableop_resource:@;
-batch_normalization_6_readvariableop_resource:@=
/batch_normalization_6_readvariableop_1_resource:@L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1бconv2d_7/BiasAdd/ReadVariableOpбconv2d_7/Conv2D/ReadVariableOpЈ
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0▒
conv2d_7/Conv2DConv2Dinput_tensor&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
ё
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:         @f
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      h
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:¤
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_7/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:           @*
half_pixel_centers(ј
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0█
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           @:@:@:@:@:*
epsilon%oЃ:*
is_training( Ђ
IdentityIdentity*batch_normalization_6/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:           @╦
NoOpNoOp6^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1 ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
═
Ю
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1904176

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
б
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1904050

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:й
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
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Г
Њ
)__inference_decoder_layer_call_fn_1904651
input_1
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ$
	unknown_7:ђ@
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
identityѕбStatefulPartitionedCallј
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
:         ђђ*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1904600y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
▓	
Љ
1__inference_decoder_block_1_layer_call_fn_1905499
input_tensor"
unknown:ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904298w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
ќ	
м
7__inference_batch_normalization_7_layer_call_fn_1905894

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1904158Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
б
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1905881

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:й
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
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Й"
љ
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905482
input_tensorC
'conv2d_6_conv2d_readvariableop_resource:ђђ7
(conv2d_6_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1бconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpљ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▓
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ё
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ў
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђk
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђd
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
:╠
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_6/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Ј
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0я
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ѓ
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђ╦
NoOpNoOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
═
Ю
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905943

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
й+
ѕ

D__inference_decoder_layer_call_and_return_conditional_losses_1904537
input_1$
dense_2_1904393:ђђђ
dense_2_1904395:
ђђ1
decoder_block_1904429:ђђ$
decoder_block_1904431:	ђ$
decoder_block_1904433:	ђ$
decoder_block_1904435:	ђ$
decoder_block_1904437:	ђ$
decoder_block_1904439:	ђ2
decoder_block_1_1904471:ђ@%
decoder_block_1_1904473:@%
decoder_block_1_1904475:@%
decoder_block_1_1904477:@%
decoder_block_1_1904479:@%
decoder_block_1_1904481:@1
decoder_block_2_1904513:@ %
decoder_block_2_1904515: %
decoder_block_2_1904517: %
decoder_block_2_1904519: %
decoder_block_2_1904521: %
decoder_block_2_1904523: *
conv2d_9_1904526: 
conv2d_9_1904528:+
conv2d_10_1904531:
conv2d_10_1904533:
identityѕб!conv2d_10/StatefulPartitionedCallб conv2d_9/StatefulPartitionedCallб%decoder_block/StatefulPartitionedCallб'decoder_block_1/StatefulPartitionedCallб'decoder_block_2/StatefulPartitionedCallбdense_2/StatefulPartitionedCallэ
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2_1904393dense_2_1904395*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1904218f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ј
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђЃ
%decoder_block/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_1904429decoder_block_1904431decoder_block_1904433decoder_block_1904435decoder_block_1904437decoder_block_1904439*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904428░
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_block/StatefulPartitionedCall:output:0decoder_block_1_1904471decoder_block_1_1904473decoder_block_1_1904475decoder_block_1_1904477decoder_block_1_1904479decoder_block_1_1904481*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904470┤
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0decoder_block_2_1904513decoder_block_2_1904515decoder_block_2_1904517decoder_block_2_1904519decoder_block_2_1904521decoder_block_2_1904523*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904512г
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_9_1904526conv2d_9_1904528*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1904366Е
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_1904531conv2d_10_1904533*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1904383Ѓ
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђФ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
ч
а
+__inference_conv2d_10_layer_call_fn_1905695

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1904383y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
љ
■
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1905686

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ђђk
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ђђw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ 
 
_user_specified_nameinputs
и+
ѕ

D__inference_decoder_layer_call_and_return_conditional_losses_1904390
input_1$
dense_2_1904219:ђђђ
dense_2_1904221:
ђђ1
decoder_block_1904256:ђђ$
decoder_block_1904258:	ђ$
decoder_block_1904260:	ђ$
decoder_block_1904262:	ђ$
decoder_block_1904264:	ђ$
decoder_block_1904266:	ђ2
decoder_block_1_1904299:ђ@%
decoder_block_1_1904301:@%
decoder_block_1_1904303:@%
decoder_block_1_1904305:@%
decoder_block_1_1904307:@%
decoder_block_1_1904309:@1
decoder_block_2_1904342:@ %
decoder_block_2_1904344: %
decoder_block_2_1904346: %
decoder_block_2_1904348: %
decoder_block_2_1904350: %
decoder_block_2_1904352: *
conv2d_9_1904367: 
conv2d_9_1904369:+
conv2d_10_1904384:
conv2d_10_1904386:
identityѕб!conv2d_10/StatefulPartitionedCallб conv2d_9/StatefulPartitionedCallб%decoder_block/StatefulPartitionedCallб'decoder_block_1/StatefulPartitionedCallб'decoder_block_2/StatefulPartitionedCallбdense_2/StatefulPartitionedCallэ
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_2_1904219dense_2_1904221*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1904218f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ј
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђЂ
%decoder_block/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_1904256decoder_block_1904258decoder_block_1904260decoder_block_1904262decoder_block_1904264decoder_block_1904266*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904255«
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_block/StatefulPartitionedCall:output:0decoder_block_1_1904299decoder_block_1_1904301decoder_block_1_1904303decoder_block_1_1904305decoder_block_1_1904307decoder_block_1_1904309*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904298▓
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0decoder_block_2_1904342decoder_block_2_1904344decoder_block_2_1904346decoder_block_2_1904348decoder_block_2_1904350decoder_block_2_1904352*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904341г
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_9_1904367conv2d_9_1904369*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1904366Е
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_1904384conv2d_10_1904386*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1904383Ѓ
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђФ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
┤	
Љ
1__inference_decoder_block_1_layer_call_fn_1905516
input_tensor"
unknown:ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904470w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
═
Ю
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905864

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
¤+
љ

D__inference_decoder_layer_call_and_return_conditional_losses_1904600
embedding_input$
dense_2_1904543:ђђђ
dense_2_1904545:
ђђ1
decoder_block_1904550:ђђ$
decoder_block_1904552:	ђ$
decoder_block_1904554:	ђ$
decoder_block_1904556:	ђ$
decoder_block_1904558:	ђ$
decoder_block_1904560:	ђ2
decoder_block_1_1904563:ђ@%
decoder_block_1_1904565:@%
decoder_block_1_1904567:@%
decoder_block_1_1904569:@%
decoder_block_1_1904571:@%
decoder_block_1_1904573:@1
decoder_block_2_1904576:@ %
decoder_block_2_1904578: %
decoder_block_2_1904580: %
decoder_block_2_1904582: %
decoder_block_2_1904584: %
decoder_block_2_1904586: *
conv2d_9_1904589: 
conv2d_9_1904591:+
conv2d_10_1904594:
conv2d_10_1904596:
identityѕб!conv2d_10/StatefulPartitionedCallб conv2d_9/StatefulPartitionedCallб%decoder_block/StatefulPartitionedCallб'decoder_block_1/StatefulPartitionedCallб'decoder_block_2/StatefulPartitionedCallбdense_2/StatefulPartitionedCall 
dense_2/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_2_1904543dense_2_1904545*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1904218f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             Ј
ReshapeReshape(dense_2/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђЂ
%decoder_block/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0decoder_block_1904550decoder_block_1904552decoder_block_1904554decoder_block_1904556decoder_block_1904558decoder_block_1904560*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_decoder_block_layer_call_and_return_conditional_losses_1904255«
'decoder_block_1/StatefulPartitionedCallStatefulPartitionedCall.decoder_block/StatefulPartitionedCall:output:0decoder_block_1_1904563decoder_block_1_1904565decoder_block_1_1904567decoder_block_1_1904569decoder_block_1_1904571decoder_block_1_1904573*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1904298▓
'decoder_block_2/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_1/StatefulPartitionedCall:output:0decoder_block_2_1904576decoder_block_2_1904578decoder_block_2_1904580decoder_block_2_1904582decoder_block_2_1904584decoder_block_2_1904586*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *U
fPRN
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904341г
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall0decoder_block_2/StatefulPartitionedCall:output:0conv2d_9_1904589conv2d_9_1904591*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1904366Е
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_1904594conv2d_10_1904596*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1904383Ѓ
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђФ
NoOpNoOp"^conv2d_10/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall&^decoder_block/StatefulPartitionedCall(^decoder_block_1/StatefulPartitionedCall(^decoder_block_2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2N
%decoder_block/StatefulPartitionedCall%decoder_block/StatefulPartitionedCall2R
'decoder_block_1/StatefulPartitionedCall'decoder_block_1/StatefulPartitionedCall2R
'decoder_block_2/StatefulPartitionedCall'decoder_block_2/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
й"
І
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1904512
input_tensorA
'conv2d_8_conv2d_readvariableop_resource:@ 6
(conv2d_8_biasadd_readvariableop_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1бconv2d_8/BiasAdd/ReadVariableOpбconv2d_8/Conv2D/ReadVariableOpј
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0▒
conv2d_8/Conv2DConv2Dinput_tensor&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
ё
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            f
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Л
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_8/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(ј
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0П
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ ╦
NoOpNoOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':           @: : : : : : 2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:           @
&
_user_specified_nameinput_tensor
щ
Ъ
*__inference_conv2d_9_layer_call_fn_1905675

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ђђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1904366y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ђђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ 
 
_user_specified_nameinputs
п)
Я
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905453
input_tensorC
'conv2d_6_conv2d_readvariableop_resource:ђђ7
(conv2d_6_biasadd_readvariableop_resource:	ђ<
-batch_normalization_5_readvariableop_resource:	ђ>
/batch_normalization_5_readvariableop_1_resource:	ђM
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:	ђO
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб$batch_normalization_5/AssignNewValueб&batch_normalization_5/AssignNewValue_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1бconv2d_6/BiasAdd/ReadVariableOpбconv2d_6/Conv2D/ReadVariableOpљ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0▓
conv2d_6/Conv2DConv2Dinput_tensor&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Ё
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ў
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђk
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђd
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
:╠
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_6/Relu:activations:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(Ј
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Њ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▒
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0х
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0В
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ѓ
IdentityIdentity*batch_normalization_5/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђЏ
NoOpNoOp%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
╗
K
/__inference_up_sampling2d_layer_call_fn_1905711

inputs
identityП
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1903967Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Є
┴
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1904158

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
а
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1903967

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:й
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
:х
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(ў
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
й"
І
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905666
input_tensorA
'conv2d_8_conv2d_readvariableop_resource:@ 6
(conv2d_8_biasadd_readvariableop_resource: ;
-batch_normalization_7_readvariableop_resource: =
/batch_normalization_7_readvariableop_1_resource: L
>batch_normalization_7_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1бconv2d_8/BiasAdd/ReadVariableOpбconv2d_8/Conv2D/ReadVariableOpј
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0▒
conv2d_8/Conv2DConv2Dinput_tensor&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:            *
paddingSAME*
strides
ё
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:            j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:            f
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"        h
up_sampling2d_2/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      Ђ
up_sampling2d_2/mulMulup_sampling2d_2/Const:output:0 up_sampling2d_2/Const_1:output:0*
T0*
_output_shapes
:Л
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_8/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:         ђђ *
half_pixel_centers(ј
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0П
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ : : : : :*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity*batch_normalization_7/FusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:         ђђ ╦
NoOpNoOp6^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':           @: : : : : : 2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:           @
&
_user_specified_nameinput_tensor
ў	
м
7__inference_batch_normalization_7_layer_call_fn_1905907

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1904176Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs"з
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Х
serving_defaultб
<
input_11
serving_default_input_1:0         ђF
output_1:
StatefulPartitionedCall:0         ђђtensorflow/serving/predict:єы
ю
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
о
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
д
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
╩
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
п
,trace_0
-trace_1
.trace_2
/trace_32ь
)__inference_decoder_layer_call_fn_1904651
)__inference_decoder_layer_call_fn_1904764
)__inference_decoder_layer_call_fn_1905113
)__inference_decoder_layer_call_fn_1905166Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 z,trace_0z-trace_1z.trace_2z/trace_3
─
0trace_0
1trace_1
2trace_2
3trace_32┘
D__inference_decoder_layer_call_and_return_conditional_losses_1904390
D__inference_decoder_layer_call_and_return_conditional_losses_1904537
D__inference_decoder_layer_call_and_return_conditional_losses_1905268
D__inference_decoder_layer_call_and_return_conditional_losses_1905370Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 z0trace_0z1trace_1z2trace_2z3trace_3
═B╩
"__inference__wrapped_model_1903954input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╗
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
┬
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
┬
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
┬
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
П
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
П
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
+:)ђђђ2decoder/dense_2/kernel
$:"ђђ2decoder/dense_2/bias
A:?ђђ2%decoder/decoder_block/conv2d_6/kernel
2:0ђ2#decoder/decoder_block/conv2d_6/bias
@:>ђ21decoder/decoder_block/batch_normalization_5/gamma
?:=ђ20decoder/decoder_block/batch_normalization_5/beta
H:Fђ (27decoder/decoder_block/batch_normalization_5/moving_mean
L:Jђ (2;decoder/decoder_block/batch_normalization_5/moving_variance
B:@ђ@2'decoder/decoder_block_1/conv2d_7/kernel
3:1@2%decoder/decoder_block_1/conv2d_7/bias
A:?@23decoder/decoder_block_1/batch_normalization_6/gamma
@:>@22decoder/decoder_block_1/batch_normalization_6/beta
I:G@ (29decoder/decoder_block_1/batch_normalization_6/moving_mean
M:K@ (2=decoder/decoder_block_1/batch_normalization_6/moving_variance
A:?@ 2'decoder/decoder_block_2/conv2d_8/kernel
3:1 2%decoder/decoder_block_2/conv2d_8/bias
A:? 23decoder/decoder_block_2/batch_normalization_7/gamma
@:> 22decoder/decoder_block_2/batch_normalization_7/beta
I:G  (29decoder/decoder_block_2/batch_normalization_7/moving_mean
M:K  (2=decoder/decoder_block_2/batch_normalization_7/moving_variance
1:/ 2decoder/conv2d_9/kernel
#:!2decoder/conv2d_9/bias
2:02decoder/conv2d_10/kernel
$:"2decoder/conv2d_10/bias
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
ЩBэ
)__inference_decoder_layer_call_fn_1904651input_1"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЩBэ
)__inference_decoder_layer_call_fn_1904764input_1"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ѓB 
)__inference_decoder_layer_call_fn_1905113embedding_input"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ѓB 
)__inference_decoder_layer_call_fn_1905166embedding_input"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЋBњ
D__inference_decoder_layer_call_and_return_conditional_losses_1904390input_1"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЋBњ
D__inference_decoder_layer_call_and_return_conditional_losses_1904537input_1"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЮBџ
D__inference_decoder_layer_call_and_return_conditional_losses_1905268embedding_input"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
ЮBџ
D__inference_decoder_layer_call_and_return_conditional_losses_1905370embedding_input"Й
и▓│
FullArgSpec
argsџ
jembedding_input
varargs
 
varkw
 
defaults
 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
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
Г
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
с
itrace_02к
)__inference_dense_2_layer_call_fn_1905379ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zitrace_0
■
jtrace_02р
D__inference_dense_2_layer_call_and_return_conditional_losses_1905390ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
Г
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
╦
ptrace_0
qtrace_12ћ
/__inference_decoder_block_layer_call_fn_1905407
/__inference_decoder_block_layer_call_fn_1905424»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zptrace_0zqtrace_1
Ђ
rtrace_0
strace_12╩
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905453
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905482»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zrtrace_0zstrace_1
П
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
д
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
ё	keras_api
Ё__call__
+є&call_and_return_all_conditional_losses
	Єaxis
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
▓
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
М
Їtrace_0
јtrace_12ў
1__inference_decoder_block_1_layer_call_fn_1905499
1__inference_decoder_block_1_layer_call_fn_1905516»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0zјtrace_1
Ѕ
Јtrace_0
љtrace_12╬
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905545
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905574»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЈtrace_0zљtrace_1
С
Љ	variables
њtrainable_variables
Њregularization_losses
ћ	keras_api
Ћ__call__
+ќ&call_and_return_all_conditional_losses

kernel
bias
!Ќ_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ў	variables
Ўtrainable_variables
џregularization_losses
Џ	keras_api
ю__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
ъ	variables
Ъtrainable_variables
аregularization_losses
А	keras_api
б__call__
+Б&call_and_return_all_conditional_losses
	цaxis
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
▓
Цnon_trainable_variables
дlayers
Дmetrics
 еlayer_regularization_losses
Еlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
М
фtrace_0
Фtrace_12ў
1__inference_decoder_block_2_layer_call_fn_1905591
1__inference_decoder_block_2_layer_call_fn_1905608»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zфtrace_0zФtrace_1
Ѕ
гtrace_0
Гtrace_12╬
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905637
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905666»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zгtrace_0zГtrace_1
С
«	variables
»trainable_variables
░regularization_losses
▒	keras_api
▓__call__
+│&call_and_return_all_conditional_losses

kernel
bias
!┤_jit_compiled_convolution_op"
_tf_keras_layer
Ф
х	variables
Хtrainable_variables
иregularization_losses
И	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
╗	variables
╝trainable_variables
йregularization_losses
Й	keras_api
┐__call__
+└&call_and_return_all_conditional_losses
	┴axis
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
▓
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
кlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
Т
Кtrace_02К
*__inference_conv2d_9_layer_call_fn_1905675ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zКtrace_0
Ђ
╚trace_02Р
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1905686ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╚trace_0
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▓
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
у
╬trace_02╚
+__inference_conv2d_10_layer_call_fn_1905695ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╬trace_0
ѓ
¤trace_02с
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1905706ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z¤trace_0
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
╠B╔
%__inference_signature_wrapper_1905060input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
МBл
)__inference_dense_2_layer_call_fn_1905379inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЬBв
D__inference_dense_2_layer_call_and_return_conditional_losses_1905390inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ШBз
/__inference_decoder_block_layer_call_fn_1905407input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
/__inference_decoder_block_layer_call_fn_1905424input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905453input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905482input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
▓
лnon_trainable_variables
Лlayers
мmetrics
 Мlayer_regularization_losses
нlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
ъ2Џў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъ2Џў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
Нnon_trainable_variables
оlayers
Оmetrics
 пlayer_regularization_losses
┘layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
в
┌trace_02╠
/__inference_up_sampling2d_layer_call_fn_1905711ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┌trace_0
є
█trace_02у
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1905723ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z█trace_0
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
И
▄non_trainable_variables
Пlayers
яmetrics
 ▀layer_regularization_losses
Яlayer_metrics
Ђ	variables
ѓtrainable_variables
Ѓregularization_losses
Ё__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
т
рtrace_0
Рtrace_12ф
7__inference_batch_normalization_5_layer_call_fn_1905736
7__inference_batch_normalization_5_layer_call_fn_1905749х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zрtrace_0zРtrace_1
Џ
сtrace_0
Сtrace_12Я
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905767
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905785х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zсtrace_0zСtrace_1
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
ЭBш
1__inference_decoder_block_1_layer_call_fn_1905499input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
1__inference_decoder_block_1_layer_call_fn_1905516input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905545input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905574input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
И
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
Љ	variables
њtrainable_variables
Њregularization_losses
Ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
ъ2Џў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъ2Џў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
ў	variables
Ўtrainable_variables
џregularization_losses
ю__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
ь
№trace_02╬
1__inference_up_sampling2d_1_layer_call_fn_1905790ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z№trace_0
ѕ
­trace_02ж
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1905802ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z­trace_0
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
И
ыnon_trainable_variables
Ыlayers
зmetrics
 Зlayer_regularization_losses
шlayer_metrics
ъ	variables
Ъtrainable_variables
аregularization_losses
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
т
Шtrace_0
эtrace_12ф
7__inference_batch_normalization_6_layer_call_fn_1905815
7__inference_batch_normalization_6_layer_call_fn_1905828х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zШtrace_0zэtrace_1
Џ
Эtrace_0
щtrace_12Я
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905846
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905864х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЭtrace_0zщtrace_1
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
ЭBш
1__inference_decoder_block_2_layer_call_fn_1905591input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
1__inference_decoder_block_2_layer_call_fn_1905608input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905637input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905666input_tensor"»
е▓ц
FullArgSpec'
argsџ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
И
Щnon_trainable_variables
чlayers
Чmetrics
 §layer_regularization_losses
■layer_metrics
«	variables
»trainable_variables
░regularization_losses
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
ъ2Џў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ъ2Џў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ф2Дц
Џ▓Ќ
FullArgSpec
argsџ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
 non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
х	variables
Хtrainable_variables
иregularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
ь
ёtrace_02╬
1__inference_up_sampling2d_2_layer_call_fn_1905869ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zёtrace_0
ѕ
Ёtrace_02ж
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1905881ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
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
И
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
╗	variables
╝trainable_variables
йregularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
т
Іtrace_0
їtrace_12ф
7__inference_batch_normalization_7_layer_call_fn_1905894
7__inference_batch_normalization_7_layer_call_fn_1905907х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zІtrace_0zїtrace_1
Џ
Їtrace_0
јtrace_12Я
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905925
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905943х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0zјtrace_1
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
нBЛ
*__inference_conv2d_9_layer_call_fn_1905675inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№BВ
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1905686inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
НBм
+__inference_conv2d_10_layer_call_fn_1905695inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­Bь
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1905706inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
┘Bо
/__inference_up_sampling2d_layer_call_fn_1905711inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЗBы
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1905723inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
■Bч
7__inference_batch_normalization_5_layer_call_fn_1905736inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
7__inference_batch_normalization_5_layer_call_fn_1905749inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905767inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905785inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
█Bп
1__inference_up_sampling2d_1_layer_call_fn_1905790inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1905802inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
■Bч
7__inference_batch_normalization_6_layer_call_fn_1905815inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
7__inference_batch_normalization_6_layer_call_fn_1905828inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905846inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905864inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
█Bп
1__inference_up_sampling2d_2_layer_call_fn_1905869inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1905881inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
■Bч
7__inference_batch_normalization_7_layer_call_fn_1905894inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
7__inference_batch_normalization_7_layer_call_fn_1905907inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905925inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905943inputs"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 │
"__inference__wrapped_model_1903954ї !"#$%&1б.
'б$
"і
input_1         ђ
ф "=ф:
8
output_1,і)
output_1         ђђЩ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905767БRбO
HбE
;і8
inputs,                           ђ
p

 
ф "GбD
=і:
tensor_0,                           ђ
џ Щ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1905785БRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "GбD
=і:
tensor_0,                           ђ
џ н
7__inference_batch_normalization_5_layer_call_fn_1905736ўRбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђн
7__inference_batch_normalization_5_layer_call_fn_1905749ўRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђЭ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905846АQбN
GбD
:і7
inputs+                           @
p

 
ф "FбC
<і9
tensor_0+                           @
џ Э
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1905864АQбN
GбD
:і7
inputs+                           @
p 

 
ф "FбC
<і9
tensor_0+                           @
џ м
7__inference_batch_normalization_6_layer_call_fn_1905815ќQбN
GбD
:і7
inputs+                           @
p

 
ф ";і8
unknown+                           @м
7__inference_batch_normalization_6_layer_call_fn_1905828ќQбN
GбD
:і7
inputs+                           @
p 

 
ф ";і8
unknown+                           @Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905925А !"QбN
GбD
:і7
inputs+                            
p

 
ф "FбC
<і9
tensor_0+                            
џ Э
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1905943А !"QбN
GбD
:і7
inputs+                            
p 

 
ф "FбC
<і9
tensor_0+                            
џ м
7__inference_batch_normalization_7_layer_call_fn_1905894ќ !"QбN
GбD
:і7
inputs+                            
p

 
ф ";і8
unknown+                            м
7__inference_batch_normalization_7_layer_call_fn_1905907ќ !"QбN
GбD
:і7
inputs+                            
p 

 
ф ";і8
unknown+                            ┴
F__inference_conv2d_10_layer_call_and_return_conditional_losses_1905706w%&9б6
/б,
*і'
inputs         ђђ
ф "6б3
,і)
tensor_0         ђђ
џ Џ
+__inference_conv2d_10_layer_call_fn_1905695l%&9б6
/б,
*і'
inputs         ђђ
ф "+і(
unknown         ђђ└
E__inference_conv2d_9_layer_call_and_return_conditional_losses_1905686w#$9б6
/б,
*і'
inputs         ђђ 
ф "6б3
,і)
tensor_0         ђђ
џ џ
*__inference_conv2d_9_layer_call_fn_1905675l#$9б6
/б,
*і'
inputs         ђђ 
ф "+і(
unknown         ђђМ
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905545ѓBб?
8б5
/і,
input_tensor         ђ
p
ф "4б1
*і'
tensor_0           @
џ М
L__inference_decoder_block_1_layer_call_and_return_conditional_losses_1905574ѓBб?
8б5
/і,
input_tensor         ђ
p 
ф "4б1
*і'
tensor_0           @
џ г
1__inference_decoder_block_1_layer_call_fn_1905499wBб?
8б5
/і,
input_tensor         ђ
p
ф ")і&
unknown           @г
1__inference_decoder_block_1_layer_call_fn_1905516wBб?
8б5
/і,
input_tensor         ђ
p 
ф ")і&
unknown           @н
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905637Ѓ !"Aб>
7б4
.і+
input_tensor           @
p
ф "6б3
,і)
tensor_0         ђђ 
џ н
L__inference_decoder_block_2_layer_call_and_return_conditional_losses_1905666Ѓ !"Aб>
7б4
.і+
input_tensor           @
p 
ф "6б3
,і)
tensor_0         ђђ 
џ Г
1__inference_decoder_block_2_layer_call_fn_1905591x !"Aб>
7б4
.і+
input_tensor           @
p
ф "+і(
unknown         ђђ Г
1__inference_decoder_block_2_layer_call_fn_1905608x !"Aб>
7б4
.і+
input_tensor           @
p 
ф "+і(
unknown         ђђ м
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905453ЃBб?
8б5
/і,
input_tensor         ђ
p
ф "5б2
+і(
tensor_0         ђ
џ м
J__inference_decoder_block_layer_call_and_return_conditional_losses_1905482ЃBб?
8б5
/і,
input_tensor         ђ
p 
ф "5б2
+і(
tensor_0         ђ
џ Ф
/__inference_decoder_block_layer_call_fn_1905407xBб?
8б5
/і,
input_tensor         ђ
p
ф "*і'
unknown         ђФ
/__inference_decoder_block_layer_call_fn_1905424xBб?
8б5
/і,
input_tensor         ђ
p 
ф "*і'
unknown         ђя
D__inference_decoder_layer_call_and_return_conditional_losses_1904390Ћ !"#$%&Aб>
'б$
"і
input_1         ђ
ф

trainingp"6б3
,і)
tensor_0         ђђ
џ я
D__inference_decoder_layer_call_and_return_conditional_losses_1904537Ћ !"#$%&Aб>
'б$
"і
input_1         ђ
ф

trainingp "6б3
,і)
tensor_0         ђђ
џ Т
D__inference_decoder_layer_call_and_return_conditional_losses_1905268Ю !"#$%&IбF
/б,
*і'
embedding_input         ђ
ф

trainingp"6б3
,і)
tensor_0         ђђ
џ Т
D__inference_decoder_layer_call_and_return_conditional_losses_1905370Ю !"#$%&IбF
/б,
*і'
embedding_input         ђ
ф

trainingp "6б3
,і)
tensor_0         ђђ
џ И
)__inference_decoder_layer_call_fn_1904651і !"#$%&Aб>
'б$
"і
input_1         ђ
ф

trainingp"+і(
unknown         ђђИ
)__inference_decoder_layer_call_fn_1904764і !"#$%&Aб>
'б$
"і
input_1         ђ
ф

trainingp "+і(
unknown         ђђ└
)__inference_decoder_layer_call_fn_1905113њ !"#$%&IбF
/б,
*і'
embedding_input         ђ
ф

trainingp"+і(
unknown         ђђ└
)__inference_decoder_layer_call_fn_1905166њ !"#$%&IбF
/б,
*і'
embedding_input         ђ
ф

trainingp "+і(
unknown         ђђ«
D__inference_dense_2_layer_call_and_return_conditional_losses_1905390f0б-
&б#
!і
inputs         ђ
ф ".б+
$і!
tensor_0         ђђ
џ ѕ
)__inference_dense_2_layer_call_fn_1905379[0б-
&б#
!і
inputs         ђ
ф "#і 
unknown         ђђ┴
%__inference_signature_wrapper_1905060Ќ !"#$%&<б9
б 
2ф/
-
input_1"і
input_1         ђ"=ф:
8
output_1,і)
output_1         ђђШ
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1905802ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ л
1__inference_up_sampling2d_1_layer_call_fn_1905790џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    Ш
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1905881ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ л
1__inference_up_sampling2d_2_layer_call_fn_1905869џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    З
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1905723ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ ╬
/__inference_up_sampling2d_layer_call_fn_1905711џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    