ЏН
Н
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

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

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
resource
,
Exp
x"T
y"T"
Ttype:

2
ћ
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
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
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

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ЭР

encoder/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameencoder/dense_21/bias
|
)encoder/dense_21/bias/Read/ReadVariableOpReadVariableOpencoder/dense_21/bias*
_output_shapes	
:*
dtype0

encoder/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *(
shared_nameencoder/dense_21/kernel

+encoder/dense_21/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_21/kernel* 
_output_shapes
:
 *
dtype0

encoder/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameencoder/dense_20/bias
|
)encoder/dense_20/bias/Read/ReadVariableOpReadVariableOpencoder/dense_20/bias*
_output_shapes	
:*
dtype0

encoder/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *(
shared_nameencoder/dense_20/kernel

+encoder/dense_20/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_20/kernel* 
_output_shapes
:
 *
dtype0
ж
?encoder/encoder_block_31/batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?encoder/encoder_block_31/batch_normalization_49/moving_variance
Я
Sencoder/encoder_block_31/batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_31/batch_normalization_49/moving_variance*
_output_shapes
:@*
dtype0
Ю
;encoder/encoder_block_31/batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;encoder/encoder_block_31/batch_normalization_49/moving_mean
Ч
Oencoder/encoder_block_31/batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_31/batch_normalization_49/moving_mean*
_output_shapes
:@*
dtype0
Р
4encoder/encoder_block_31/batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64encoder/encoder_block_31/batch_normalization_49/beta
Й
Hencoder/encoder_block_31/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_31/batch_normalization_49/beta*
_output_shapes
:@*
dtype0
Т
5encoder/encoder_block_31/batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75encoder/encoder_block_31/batch_normalization_49/gamma
Л
Iencoder/encoder_block_31/batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_31/batch_normalization_49/gamma*
_output_shapes
:@*
dtype0
І
'encoder/encoder_block_31/conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'encoder/encoder_block_31/conv2d_61/bias

;encoder/encoder_block_31/conv2d_61/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_31/conv2d_61/bias*
_output_shapes
:@*
dtype0
З
)encoder/encoder_block_31/conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)encoder/encoder_block_31/conv2d_61/kernel
А
=encoder/encoder_block_31/conv2d_61/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_31/conv2d_61/kernel*'
_output_shapes
:@*
dtype0
з
?encoder/encoder_block_30/batch_normalization_48/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?encoder/encoder_block_30/batch_normalization_48/moving_variance
а
Sencoder/encoder_block_30/batch_normalization_48/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_30/batch_normalization_48/moving_variance*
_output_shapes	
:*
dtype0
Я
;encoder/encoder_block_30/batch_normalization_48/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;encoder/encoder_block_30/batch_normalization_48/moving_mean
Ш
Oencoder/encoder_block_30/batch_normalization_48/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_30/batch_normalization_48/moving_mean*
_output_shapes	
:*
dtype0
С
4encoder/encoder_block_30/batch_normalization_48/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64encoder/encoder_block_30/batch_normalization_48/beta
К
Hencoder/encoder_block_30/batch_normalization_48/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_30/batch_normalization_48/beta*
_output_shapes	
:*
dtype0
У
5encoder/encoder_block_30/batch_normalization_48/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75encoder/encoder_block_30/batch_normalization_48/gamma
М
Iencoder/encoder_block_30/batch_normalization_48/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_30/batch_normalization_48/gamma*
_output_shapes	
:*
dtype0
Ї
'encoder/encoder_block_30/conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'encoder/encoder_block_30/conv2d_60/bias
 
;encoder/encoder_block_30/conv2d_60/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_30/conv2d_60/bias*
_output_shapes	
:*
dtype0
И
)encoder/encoder_block_30/conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)encoder/encoder_block_30/conv2d_60/kernel
Б
=encoder/encoder_block_30/conv2d_60/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_30/conv2d_60/kernel*(
_output_shapes
:*
dtype0
з
?encoder/encoder_block_29/batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?encoder/encoder_block_29/batch_normalization_47/moving_variance
а
Sencoder/encoder_block_29/batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_29/batch_normalization_47/moving_variance*
_output_shapes	
:*
dtype0
Я
;encoder/encoder_block_29/batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;encoder/encoder_block_29/batch_normalization_47/moving_mean
Ш
Oencoder/encoder_block_29/batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_29/batch_normalization_47/moving_mean*
_output_shapes	
:*
dtype0
С
4encoder/encoder_block_29/batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64encoder/encoder_block_29/batch_normalization_47/beta
К
Hencoder/encoder_block_29/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_29/batch_normalization_47/beta*
_output_shapes	
:*
dtype0
У
5encoder/encoder_block_29/batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75encoder/encoder_block_29/batch_normalization_47/gamma
М
Iencoder/encoder_block_29/batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_29/batch_normalization_47/gamma*
_output_shapes	
:*
dtype0
Ї
'encoder/encoder_block_29/conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'encoder/encoder_block_29/conv2d_59/bias
 
;encoder/encoder_block_29/conv2d_59/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_29/conv2d_59/bias*
_output_shapes	
:*
dtype0
И
)encoder/encoder_block_29/conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)encoder/encoder_block_29/conv2d_59/kernel
Б
=encoder/encoder_block_29/conv2d_59/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_29/conv2d_59/kernel*(
_output_shapes
:*
dtype0
з
?encoder/encoder_block_28/batch_normalization_46/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*P
shared_nameA?encoder/encoder_block_28/batch_normalization_46/moving_variance
а
Sencoder/encoder_block_28/batch_normalization_46/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_28/batch_normalization_46/moving_variance*
_output_shapes	
:*
dtype0
Я
;encoder/encoder_block_28/batch_normalization_46/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*L
shared_name=;encoder/encoder_block_28/batch_normalization_46/moving_mean
Ш
Oencoder/encoder_block_28/batch_normalization_46/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_28/batch_normalization_46/moving_mean*
_output_shapes	
:*
dtype0
С
4encoder/encoder_block_28/batch_normalization_46/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64encoder/encoder_block_28/batch_normalization_46/beta
К
Hencoder/encoder_block_28/batch_normalization_46/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_28/batch_normalization_46/beta*
_output_shapes	
:*
dtype0
У
5encoder/encoder_block_28/batch_normalization_46/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75encoder/encoder_block_28/batch_normalization_46/gamma
М
Iencoder/encoder_block_28/batch_normalization_46/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_28/batch_normalization_46/gamma*
_output_shapes	
:*
dtype0
Ї
'encoder/encoder_block_28/conv2d_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'encoder/encoder_block_28/conv2d_58/bias
 
;encoder/encoder_block_28/conv2d_58/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_28/conv2d_58/bias*
_output_shapes	
:*
dtype0
З
)encoder/encoder_block_28/conv2d_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)encoder/encoder_block_28/conv2d_58/kernel
А
=encoder/encoder_block_28/conv2d_58/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_28/conv2d_58/kernel*'
_output_shapes
:*
dtype0

serving_default_input_1Placeholder*1
_output_shapes
:џџџџџџџџџ*
dtype0*&
shape:џџџџџџџџџ
ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)encoder/encoder_block_28/conv2d_58/kernel'encoder/encoder_block_28/conv2d_58/bias5encoder/encoder_block_28/batch_normalization_46/gamma4encoder/encoder_block_28/batch_normalization_46/beta;encoder/encoder_block_28/batch_normalization_46/moving_mean?encoder/encoder_block_28/batch_normalization_46/moving_variance)encoder/encoder_block_29/conv2d_59/kernel'encoder/encoder_block_29/conv2d_59/bias5encoder/encoder_block_29/batch_normalization_47/gamma4encoder/encoder_block_29/batch_normalization_47/beta;encoder/encoder_block_29/batch_normalization_47/moving_mean?encoder/encoder_block_29/batch_normalization_47/moving_variance)encoder/encoder_block_30/conv2d_60/kernel'encoder/encoder_block_30/conv2d_60/bias5encoder/encoder_block_30/batch_normalization_48/gamma4encoder/encoder_block_30/batch_normalization_48/beta;encoder/encoder_block_30/batch_normalization_48/moving_mean?encoder/encoder_block_30/batch_normalization_48/moving_variance)encoder/encoder_block_31/conv2d_61/kernel'encoder/encoder_block_31/conv2d_61/bias5encoder/encoder_block_31/batch_normalization_49/gamma4encoder/encoder_block_31/batch_normalization_49/beta;encoder/encoder_block_31/batch_normalization_49/moving_mean?encoder/encoder_block_31/batch_normalization_49/moving_varianceencoder/dense_20/kernelencoder/dense_20/biasencoder/dense_21/kernelencoder/dense_21/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *.
f)R'
%__inference_signature_wrapper_1935046

NoOpNoOp
Ј
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*т
valueзBг BЫ
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

block1

	block2


block3

block4

flattening

z_mean
z_logvar
	embedding

signatures*
к
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22
(23
)24
*25
+26
,27*

0
1
2
3
4
5
6
7
8
9
10
 11
#12
$13
%14
&15
)16
*17
+18
,19*
* 
А
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
2trace_0
3trace_1
4trace_2
5trace_3* 
6
6trace_0
7trace_1
8trace_2
9trace_3* 
* 
Џ
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@conv
Apooling
Bbn*
Џ
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jpooling
Kbn*
Џ
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rconv
Spooling
Tbn*
Џ
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[conv
\pooling
]bn*

^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
І
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

)kernel
*bias*
І
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

+kernel
,bias*

p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 

vserving_default* 
ic
VARIABLE_VALUE)encoder/encoder_block_28/conv2d_58/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'encoder/encoder_block_28/conv2d_58/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoder/encoder_block_28/batch_normalization_46/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4encoder/encoder_block_28/batch_normalization_46/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;encoder/encoder_block_28/batch_normalization_46/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?encoder/encoder_block_28/batch_normalization_46/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)encoder/encoder_block_29/conv2d_59/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'encoder/encoder_block_29/conv2d_59/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoder/encoder_block_29/batch_normalization_47/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4encoder/encoder_block_29/batch_normalization_47/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_29/batch_normalization_47/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE?encoder/encoder_block_29/batch_normalization_47/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)encoder/encoder_block_30/conv2d_60/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'encoder/encoder_block_30/conv2d_60/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5encoder/encoder_block_30/batch_normalization_48/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4encoder/encoder_block_30/batch_normalization_48/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_30/batch_normalization_48/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE?encoder/encoder_block_30/batch_normalization_48/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)encoder/encoder_block_31/conv2d_61/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'encoder/encoder_block_31/conv2d_61/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5encoder/encoder_block_31/batch_normalization_49/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4encoder/encoder_block_31/batch_normalization_49/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_31/batch_normalization_49/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE?encoder/encoder_block_31/batch_normalization_49/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEencoder/dense_20/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEencoder/dense_20/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEencoder/dense_21/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEencoder/dense_21/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
<
0
1
2
3
!4
"5
'6
(7*
<
0
	1

2
3
4
5
6
7*
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

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

|trace_0
}trace_1* 

~trace_0
trace_1* 
Я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
м
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
Я
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses

kernel
bias
!Ѓ_jit_compiled_convolution_op*

Є	variables
Ѕtrainable_variables
Іregularization_losses
Ї	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses* 
м
Њ	variables
Ћtrainable_variables
Ќregularization_losses
­	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses
	Аaxis
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

Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

Жtrace_0
Зtrace_1* 

Иtrace_0
Йtrace_1* 
Я
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses

kernel
bias
!Р_jit_compiled_convolution_op*

С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses* 
м
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
	Эaxis
	gamma
 beta
!moving_mean
"moving_variance*
.
#0
$1
%2
&3
'4
(5*
 
#0
$1
%2
&3*
* 

Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

гtrace_0
дtrace_1* 

еtrace_0
жtrace_1* 
Я
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses

#kernel
$bias
!н_jit_compiled_convolution_op*

о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses* 
м
ф	variables
хtrainable_variables
цregularization_losses
ч	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses
	ъaxis
	%gamma
&beta
'moving_mean
(moving_variance*
* 
* 
* 

ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

№trace_0* 

ёtrace_0* 

)0
*1*

)0
*1*
* 

ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

їtrace_0* 

јtrace_0* 

+0
,1*

+0
,1*
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

ўtrace_0* 

џtrace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
 
0
1
2
3*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses* 

Іtrace_0* 

Їtrace_0* 
 
0
1
2
3*

0
1*
* 

Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
Њ	variables
Ћtrainable_variables
Ќregularization_losses
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses*

­trace_0
Ўtrace_1* 

Џtrace_0
Аtrace_1* 
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

Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

Лtrace_0* 

Мtrace_0* 
 
0
 1
!2
"3*

0
 1*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*

Тtrace_0
Уtrace_1* 

Фtrace_0
Хtrace_1* 
* 

'0
(1*

[0
\1
]2*
* 
* 
* 
* 
* 
* 
* 

#0
$1*

#0
$1*
* 

Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses* 

аtrace_0* 

бtrace_0* 
 
%0
&1
'2
(3*

%0
&1*
* 

вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses*

зtrace_0
иtrace_1* 

йtrace_0
кtrace_1* 
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
'0
(1*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)encoder/encoder_block_28/conv2d_58/kernel'encoder/encoder_block_28/conv2d_58/bias5encoder/encoder_block_28/batch_normalization_46/gamma4encoder/encoder_block_28/batch_normalization_46/beta;encoder/encoder_block_28/batch_normalization_46/moving_mean?encoder/encoder_block_28/batch_normalization_46/moving_variance)encoder/encoder_block_29/conv2d_59/kernel'encoder/encoder_block_29/conv2d_59/bias5encoder/encoder_block_29/batch_normalization_47/gamma4encoder/encoder_block_29/batch_normalization_47/beta;encoder/encoder_block_29/batch_normalization_47/moving_mean?encoder/encoder_block_29/batch_normalization_47/moving_variance)encoder/encoder_block_30/conv2d_60/kernel'encoder/encoder_block_30/conv2d_60/bias5encoder/encoder_block_30/batch_normalization_48/gamma4encoder/encoder_block_30/batch_normalization_48/beta;encoder/encoder_block_30/batch_normalization_48/moving_mean?encoder/encoder_block_30/batch_normalization_48/moving_variance)encoder/encoder_block_31/conv2d_61/kernel'encoder/encoder_block_31/conv2d_61/bias5encoder/encoder_block_31/batch_normalization_49/gamma4encoder/encoder_block_31/batch_normalization_49/beta;encoder/encoder_block_31/batch_normalization_49/moving_mean?encoder/encoder_block_31/batch_normalization_49/moving_varianceencoder/dense_20/kernelencoder/dense_20/biasencoder/dense_21/kernelencoder/dense_21/biasConst*)
Tin"
 2*
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
GPU2 *0J 8 *)
f$R"
 __inference__traced_save_1936346

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)encoder/encoder_block_28/conv2d_58/kernel'encoder/encoder_block_28/conv2d_58/bias5encoder/encoder_block_28/batch_normalization_46/gamma4encoder/encoder_block_28/batch_normalization_46/beta;encoder/encoder_block_28/batch_normalization_46/moving_mean?encoder/encoder_block_28/batch_normalization_46/moving_variance)encoder/encoder_block_29/conv2d_59/kernel'encoder/encoder_block_29/conv2d_59/bias5encoder/encoder_block_29/batch_normalization_47/gamma4encoder/encoder_block_29/batch_normalization_47/beta;encoder/encoder_block_29/batch_normalization_47/moving_mean?encoder/encoder_block_29/batch_normalization_47/moving_variance)encoder/encoder_block_30/conv2d_60/kernel'encoder/encoder_block_30/conv2d_60/bias5encoder/encoder_block_30/batch_normalization_48/gamma4encoder/encoder_block_30/batch_normalization_48/beta;encoder/encoder_block_30/batch_normalization_48/moving_mean?encoder/encoder_block_30/batch_normalization_48/moving_variance)encoder/encoder_block_31/conv2d_61/kernel'encoder/encoder_block_31/conv2d_61/bias5encoder/encoder_block_31/batch_normalization_49/gamma4encoder/encoder_block_31/batch_normalization_49/beta;encoder/encoder_block_31/batch_normalization_49/moving_mean?encoder/encoder_block_31/batch_normalization_49/moving_varianceencoder/dense_20/kernelencoder/dense_20/biasencoder/dense_21/kernelencoder/dense_21/bias*(
Tin!
2*
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
GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_1936440ЁФ
Ї7
Г
D__inference_encoder_layer_call_and_return_conditional_losses_1934478
tensor_input3
encoder_block_28_1934411:'
encoder_block_28_1934413:	'
encoder_block_28_1934415:	'
encoder_block_28_1934417:	'
encoder_block_28_1934419:	'
encoder_block_28_1934421:	4
encoder_block_29_1934424:'
encoder_block_29_1934426:	'
encoder_block_29_1934428:	'
encoder_block_29_1934430:	'
encoder_block_29_1934432:	'
encoder_block_29_1934434:	4
encoder_block_30_1934437:'
encoder_block_30_1934439:	'
encoder_block_30_1934441:	'
encoder_block_30_1934443:	'
encoder_block_30_1934445:	'
encoder_block_30_1934447:	3
encoder_block_31_1934450:@&
encoder_block_31_1934452:@&
encoder_block_31_1934454:@&
encoder_block_31_1934456:@&
encoder_block_31_1934458:@&
encoder_block_31_1934460:@$
dense_20_1934464:
 
dense_20_1934466:	$
dense_21_1934469:
 
dense_21_1934471:	
identity

identity_1

identity_2Ђ dense_20/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ(encoder_block_28/StatefulPartitionedCallЂ(encoder_block_29/StatefulPartitionedCallЂ(encoder_block_30/StatefulPartitionedCallЂ(encoder_block_31/StatefulPartitionedCallЂ"sampling_7/StatefulPartitionedCall
(encoder_block_28/StatefulPartitionedCallStatefulPartitionedCalltensor_inputencoder_block_28_1934411encoder_block_28_1934413encoder_block_28_1934415encoder_block_28_1934417encoder_block_28_1934419encoder_block_28_1934421*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934024К
(encoder_block_29/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_28/StatefulPartitionedCall:output:0encoder_block_29_1934424encoder_block_29_1934426encoder_block_29_1934428encoder_block_29_1934430encoder_block_29_1934432encoder_block_29_1934434*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934064К
(encoder_block_30/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_29/StatefulPartitionedCall:output:0encoder_block_30_1934437encoder_block_30_1934439encoder_block_30_1934441encoder_block_30_1934443encoder_block_30_1934445encoder_block_30_1934447*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934104Й
(encoder_block_31/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_30/StatefulPartitionedCall:output:0encoder_block_31_1934450encoder_block_31_1934452encoder_block_31_1934454encoder_block_31_1934456encoder_block_31_1934458encoder_block_31_1934460*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934144ь
flatten_7/PartitionedCallPartitionedCall1encoder_block_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1934164
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_20_1934464dense_20_1934466*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1934177
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_1934469dense_21_1934471*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1934194Ђ
"sampling_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_sampling_7_layer_call_and_return_conditional_losses_1934226y
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ{

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ}

Identity_2Identity+sampling_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџн
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_28/StatefulPartitionedCall)^encoder_block_29/StatefulPartitionedCall)^encoder_block_30/StatefulPartitionedCall)^encoder_block_31/StatefulPartitionedCall#^sampling_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_28/StatefulPartitionedCall(encoder_block_28/StatefulPartitionedCall2T
(encoder_block_29/StatefulPartitionedCall(encoder_block_29/StatefulPartitionedCall2T
(encoder_block_30/StatefulPartitionedCall(encoder_block_30/StatefulPartitionedCall2T
(encoder_block_31/StatefulPartitionedCall(encoder_block_31/StatefulPartitionedCall2H
"sampling_7/StatefulPartitionedCall"sampling_7/StatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nametensor_input
й
Ё
)__inference_encoder_layer_call_fn_1934676
input_1"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
 

unknown_24:	

unknown_25:
 

unknown_26:	
identity

identity_1

identity_2ЂStatefulPartitionedCallэ
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1934613p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
С
N
2__inference_max_pooling2d_31_layer_call_fn_1936086

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_1933924
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_1933924

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Ђ
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935937

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ю

S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936153

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
 	
з
8__inference_batch_normalization_48_layer_call_fn_1936032

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1933873
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ	
з
8__inference_batch_normalization_46_layer_call_fn_1935901

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1933739
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1935991

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І'
№
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935498
input_tensorC
(conv2d_58_conv2d_readvariableop_resource:8
)conv2d_58_biasadd_readvariableop_resource:	=
.batch_normalization_46_readvariableop_resource:	?
0batch_normalization_46_readvariableop_1_resource:	N
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ%batch_normalization_46/AssignNewValueЂ'batch_normalization_46/AssignNewValue_1Ђ6batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_46/ReadVariableOpЂ'batch_normalization_46/ReadVariableOp_1Ђ conv2d_58/BiasAdd/ReadVariableOpЂconv2d_58/Conv2D/ReadVariableOp
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Ж
conv2d_58/Conv2DConv2Dinput_tensor'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџo
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџЏ
max_pooling2d_28/MaxPoolMaxPoolconv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides

%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_28/MaxPool:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_46/AssignNewValueAssignVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource4batch_normalization_46/FusedBatchNormV3:batch_mean:07^batch_normalization_46/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_46/AssignNewValue_1AssignVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_46/FusedBatchNormV3:batch_variance:09^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_46/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@@Ѓ
NoOpNoOp&^batch_normalization_46/AssignNewValue(^batch_normalization_46/AssignNewValue_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : 2R
'batch_normalization_46/AssignNewValue_1'batch_normalization_46/AssignNewValue_12N
%batch_normalization_46/AssignNewValue%batch_normalization_46/AssignNewValue2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor

Ц
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1933721

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
7
Ў
D__inference_encoder_layer_call_and_return_conditional_losses_1934231
input_13
encoder_block_28_1934025:'
encoder_block_28_1934027:	'
encoder_block_28_1934029:	'
encoder_block_28_1934031:	'
encoder_block_28_1934033:	'
encoder_block_28_1934035:	4
encoder_block_29_1934065:'
encoder_block_29_1934067:	'
encoder_block_29_1934069:	'
encoder_block_29_1934071:	'
encoder_block_29_1934073:	'
encoder_block_29_1934075:	4
encoder_block_30_1934105:'
encoder_block_30_1934107:	'
encoder_block_30_1934109:	'
encoder_block_30_1934111:	'
encoder_block_30_1934113:	'
encoder_block_30_1934115:	3
encoder_block_31_1934145:@&
encoder_block_31_1934147:@&
encoder_block_31_1934149:@&
encoder_block_31_1934151:@&
encoder_block_31_1934153:@&
encoder_block_31_1934155:@$
dense_20_1934178:
 
dense_20_1934180:	$
dense_21_1934195:
 
dense_21_1934197:	
identity

identity_1

identity_2Ђ dense_20/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ(encoder_block_28/StatefulPartitionedCallЂ(encoder_block_29/StatefulPartitionedCallЂ(encoder_block_30/StatefulPartitionedCallЂ(encoder_block_31/StatefulPartitionedCallЂ"sampling_7/StatefulPartitionedCall
(encoder_block_28/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_block_28_1934025encoder_block_28_1934027encoder_block_28_1934029encoder_block_28_1934031encoder_block_28_1934033encoder_block_28_1934035*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934024К
(encoder_block_29/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_28/StatefulPartitionedCall:output:0encoder_block_29_1934065encoder_block_29_1934067encoder_block_29_1934069encoder_block_29_1934071encoder_block_29_1934073encoder_block_29_1934075*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934064К
(encoder_block_30/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_29/StatefulPartitionedCall:output:0encoder_block_30_1934105encoder_block_30_1934107encoder_block_30_1934109encoder_block_30_1934111encoder_block_30_1934113encoder_block_30_1934115*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934104Й
(encoder_block_31/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_30/StatefulPartitionedCall:output:0encoder_block_31_1934145encoder_block_31_1934147encoder_block_31_1934149encoder_block_31_1934151encoder_block_31_1934153encoder_block_31_1934155*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934144ь
flatten_7/PartitionedCallPartitionedCall1encoder_block_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1934164
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_20_1934178dense_20_1934180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1934177
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_1934195dense_21_1934197*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1934194Ђ
"sampling_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_sampling_7_layer_call_and_return_conditional_losses_1934226y
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ{

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ}

Identity_2Identity+sampling_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџн
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_28/StatefulPartitionedCall)^encoder_block_29/StatefulPartitionedCall)^encoder_block_30/StatefulPartitionedCall)^encoder_block_31/StatefulPartitionedCall#^sampling_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_28/StatefulPartitionedCall(encoder_block_28/StatefulPartitionedCall2T
(encoder_block_29/StatefulPartitionedCall(encoder_block_29/StatefulPartitionedCall2T
(encoder_block_30/StatefulPartitionedCall(encoder_block_30/StatefulPartitionedCall2T
(encoder_block_31/StatefulPartitionedCall(encoder_block_31/StatefulPartitionedCall2H
"sampling_7/StatefulPartitionedCall"sampling_7/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ж	

2__inference_encoder_block_31_layer_call_fn_1935730
input_tensor"
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934376w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
і

M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935696
input_tensorD
(conv2d_60_conv2d_readvariableop_resource:8
)conv2d_60_biasadd_readvariableop_resource:	=
.batch_normalization_48_readvariableop_resource:	?
0batch_normalization_48_readvariableop_1_resource:	N
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ6batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_48/ReadVariableOpЂ'batch_normalization_48/ReadVariableOp_1Ђ conv2d_60/BiasAdd/ReadVariableOpЂconv2d_60/Conv2D/ReadVariableOp
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_60/Conv2DConv2Dinput_tensor'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  m
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  Џ
max_pooling2d_30/MaxPoolMaxPoolconv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_30/MaxPool:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_48/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџб
NoOpNoOp7^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ  : : : : : : 2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameinput_tensor

i
M__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_1936019

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 '
ё
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935670
input_tensorD
(conv2d_60_conv2d_readvariableop_resource:8
)conv2d_60_biasadd_readvariableop_resource:	=
.batch_normalization_48_readvariableop_resource:	?
0batch_normalization_48_readvariableop_1_resource:	N
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ%batch_normalization_48/AssignNewValueЂ'batch_normalization_48/AssignNewValue_1Ђ6batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_48/ReadVariableOpЂ'batch_normalization_48/ReadVariableOp_1Ђ conv2d_60/BiasAdd/ReadVariableOpЂconv2d_60/Conv2D/ReadVariableOp
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_60/Conv2DConv2Dinput_tensor'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  m
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  Џ
max_pooling2d_30/MaxPoolMaxPoolconv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_30/MaxPool:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_48/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЃ
NoOpNoOp&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ  : : : : : : 2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameinput_tensor

i
M__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_1936091

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д	

2__inference_encoder_block_31_layer_call_fn_1935713
input_tensor"
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityЂStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934144w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
о
Ђ
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1936009

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

љ
E__inference_dense_20_layer_call_and_return_conditional_losses_1935813

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
р

M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934376
input_tensorC
(conv2d_61_conv2d_readvariableop_resource:@7
)conv2d_61_biasadd_readvariableop_resource:@<
.batch_normalization_49_readvariableop_resource:@>
0batch_normalization_49_readvariableop_1_resource:@M
?batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@
identityЂ6batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_49/ReadVariableOpЂ'batch_normalization_49/ReadVariableOp_1Ђ conv2d_61/BiasAdd/ReadVariableOpЂconv2d_61/Conv2D/ReadVariableOp
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Г
conv2d_61/Conv2DConv2Dinput_tensor'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@l
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ў
max_pooling2d_31/MaxPoolMaxPoolconv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_31/MaxPool:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_49/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@б
NoOpNoOp7^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : 2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
С
N
2__inference_max_pooling2d_28_layer_call_fn_1935870

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1933696
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
С
N
2__inference_max_pooling2d_30_layer_call_fn_1936014

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_1933848
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1933772

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
а

*__inference_dense_20_layer_call_fn_1935802

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1934177p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ј

љ
E__inference_dense_21_layer_call_and_return_conditional_losses_1934194

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ш
І
)__inference_encoder_layer_call_fn_1935176
tensor_input"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
 

unknown_24:	

unknown_25:
 

unknown_26:	
identity

identity_1

identity_2ЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCalltensor_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1934613p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nametensor_input
Ј

љ
E__inference_dense_20_layer_call_and_return_conditional_losses_1934177

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

u
,__inference_sampling_7_layer_call_fn_1935839
inputs_0
inputs_1
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_sampling_7_layer_call_and_return_conditional_losses_1934226p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Ер
Ѕ 
 __inference__traced_save_1936346
file_prefix[
@read_disablecopyonread_encoder_encoder_block_28_conv2d_58_kernel:O
@read_1_disablecopyonread_encoder_encoder_block_28_conv2d_58_bias:	]
Nread_2_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_gamma:	\
Mread_3_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_beta:	c
Tread_4_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_moving_mean:	g
Xread_5_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_moving_variance:	^
Bread_6_disablecopyonread_encoder_encoder_block_29_conv2d_59_kernel:O
@read_7_disablecopyonread_encoder_encoder_block_29_conv2d_59_bias:	]
Nread_8_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_gamma:	\
Mread_9_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_beta:	d
Uread_10_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_moving_mean:	h
Yread_11_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_moving_variance:	_
Cread_12_disablecopyonread_encoder_encoder_block_30_conv2d_60_kernel:P
Aread_13_disablecopyonread_encoder_encoder_block_30_conv2d_60_bias:	^
Oread_14_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_gamma:	]
Nread_15_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_beta:	d
Uread_16_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_moving_mean:	h
Yread_17_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_moving_variance:	^
Cread_18_disablecopyonread_encoder_encoder_block_31_conv2d_61_kernel:@O
Aread_19_disablecopyonread_encoder_encoder_block_31_conv2d_61_bias:@]
Oread_20_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_gamma:@\
Nread_21_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_beta:@c
Uread_22_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_moving_mean:@g
Yread_23_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_moving_variance:@E
1read_24_disablecopyonread_encoder_dense_20_kernel:
 >
/read_25_disablecopyonread_encoder_dense_20_bias:	E
1read_26_disablecopyonread_encoder_dense_21_kernel:
 >
/read_27_disablecopyonread_encoder_dense_21_bias:	
savev2_const
identity_57ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead@read_disablecopyonread_encoder_encoder_block_28_conv2d_58_kernel"/device:CPU:0*
_output_shapes
 Х
Read/ReadVariableOpReadVariableOp@read_disablecopyonread_encoder_encoder_block_28_conv2d_58_kernel^Read/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:*
dtype0r
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:j

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*'
_output_shapes
:
Read_1/DisableCopyOnReadDisableCopyOnRead@read_1_disablecopyonread_encoder_encoder_block_28_conv2d_58_bias"/device:CPU:0*
_output_shapes
 Н
Read_1/ReadVariableOpReadVariableOp@read_1_disablecopyonread_encoder_encoder_block_28_conv2d_58_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ђ
Read_2/DisableCopyOnReadDisableCopyOnReadNread_2_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_gamma"/device:CPU:0*
_output_shapes
 Ы
Read_2/ReadVariableOpReadVariableOpNread_2_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ё
Read_3/DisableCopyOnReadDisableCopyOnReadMread_3_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_beta"/device:CPU:0*
_output_shapes
 Ъ
Read_3/ReadVariableOpReadVariableOpMread_3_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ј
Read_4/DisableCopyOnReadDisableCopyOnReadTread_4_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_moving_mean"/device:CPU:0*
_output_shapes
 б
Read_4/ReadVariableOpReadVariableOpTread_4_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ќ
Read_5/DisableCopyOnReadDisableCopyOnReadXread_5_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_moving_variance"/device:CPU:0*
_output_shapes
 е
Read_5/ReadVariableOpReadVariableOpXread_5_disablecopyonread_encoder_encoder_block_28_batch_normalization_46_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_6/DisableCopyOnReadDisableCopyOnReadBread_6_disablecopyonread_encoder_encoder_block_29_conv2d_59_kernel"/device:CPU:0*
_output_shapes
 Ь
Read_6/ReadVariableOpReadVariableOpBread_6_disablecopyonread_encoder_encoder_block_29_conv2d_59_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:*
dtype0x
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:o
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*(
_output_shapes
:
Read_7/DisableCopyOnReadDisableCopyOnRead@read_7_disablecopyonread_encoder_encoder_block_29_conv2d_59_bias"/device:CPU:0*
_output_shapes
 Н
Read_7/ReadVariableOpReadVariableOp@read_7_disablecopyonread_encoder_encoder_block_29_conv2d_59_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ђ
Read_8/DisableCopyOnReadDisableCopyOnReadNread_8_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_gamma"/device:CPU:0*
_output_shapes
 Ы
Read_8/ReadVariableOpReadVariableOpNread_8_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ё
Read_9/DisableCopyOnReadDisableCopyOnReadMread_9_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_beta"/device:CPU:0*
_output_shapes
 Ъ
Read_9/ReadVariableOpReadVariableOpMread_9_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:Њ
Read_10/DisableCopyOnReadDisableCopyOnReadUread_10_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_moving_mean"/device:CPU:0*
_output_shapes
 д
Read_10/ReadVariableOpReadVariableOpUread_10_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ў
Read_11/DisableCopyOnReadDisableCopyOnReadYread_11_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_moving_variance"/device:CPU:0*
_output_shapes
 и
Read_11/ReadVariableOpReadVariableOpYread_11_disablecopyonread_encoder_encoder_block_29_batch_normalization_47_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_12/DisableCopyOnReadDisableCopyOnReadCread_12_disablecopyonread_encoder_encoder_block_30_conv2d_60_kernel"/device:CPU:0*
_output_shapes
 Я
Read_12/ReadVariableOpReadVariableOpCread_12_disablecopyonread_encoder_encoder_block_30_conv2d_60_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:o
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:
Read_13/DisableCopyOnReadDisableCopyOnReadAread_13_disablecopyonread_encoder_encoder_block_30_conv2d_60_bias"/device:CPU:0*
_output_shapes
 Р
Read_13/ReadVariableOpReadVariableOpAread_13_disablecopyonread_encoder_encoder_block_30_conv2d_60_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:Є
Read_14/DisableCopyOnReadDisableCopyOnReadOread_14_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_gamma"/device:CPU:0*
_output_shapes
 Ю
Read_14/ReadVariableOpReadVariableOpOread_14_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ѓ
Read_15/DisableCopyOnReadDisableCopyOnReadNread_15_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_beta"/device:CPU:0*
_output_shapes
 Э
Read_15/ReadVariableOpReadVariableOpNread_15_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:Њ
Read_16/DisableCopyOnReadDisableCopyOnReadUread_16_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_moving_mean"/device:CPU:0*
_output_shapes
 д
Read_16/ReadVariableOpReadVariableOpUread_16_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ў
Read_17/DisableCopyOnReadDisableCopyOnReadYread_17_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_moving_variance"/device:CPU:0*
_output_shapes
 и
Read_17/ReadVariableOpReadVariableOpYread_17_disablecopyonread_encoder_encoder_block_30_batch_normalization_48_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_18/DisableCopyOnReadDisableCopyOnReadCread_18_disablecopyonread_encoder_encoder_block_31_conv2d_61_kernel"/device:CPU:0*
_output_shapes
 Ю
Read_18/ReadVariableOpReadVariableOpCread_18_disablecopyonread_encoder_encoder_block_31_conv2d_61_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@*
dtype0x
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@n
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*'
_output_shapes
:@
Read_19/DisableCopyOnReadDisableCopyOnReadAread_19_disablecopyonread_encoder_encoder_block_31_conv2d_61_bias"/device:CPU:0*
_output_shapes
 П
Read_19/ReadVariableOpReadVariableOpAread_19_disablecopyonread_encoder_encoder_block_31_conv2d_61_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@Є
Read_20/DisableCopyOnReadDisableCopyOnReadOread_20_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_gamma"/device:CPU:0*
_output_shapes
 Э
Read_20/ReadVariableOpReadVariableOpOread_20_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ѓ
Read_21/DisableCopyOnReadDisableCopyOnReadNread_21_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_beta"/device:CPU:0*
_output_shapes
 Ь
Read_21/ReadVariableOpReadVariableOpNread_21_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
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
:@Њ
Read_22/DisableCopyOnReadDisableCopyOnReadUread_22_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_moving_mean"/device:CPU:0*
_output_shapes
 г
Read_22/ReadVariableOpReadVariableOpUread_22_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ў
Read_23/DisableCopyOnReadDisableCopyOnReadYread_23_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_moving_variance"/device:CPU:0*
_output_shapes
 з
Read_23/ReadVariableOpReadVariableOpYread_23_disablecopyonread_encoder_encoder_block_31_batch_normalization_49_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_24/DisableCopyOnReadDisableCopyOnRead1read_24_disablecopyonread_encoder_dense_20_kernel"/device:CPU:0*
_output_shapes
 Е
Read_24/ReadVariableOpReadVariableOp1read_24_disablecopyonread_encoder_dense_20_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_encoder_dense_20_bias"/device:CPU:0*
_output_shapes
 Ў
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_encoder_dense_20_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_26/DisableCopyOnReadDisableCopyOnRead1read_26_disablecopyonread_encoder_dense_21_kernel"/device:CPU:0*
_output_shapes
 Е
Read_26/ReadVariableOpReadVariableOp1read_26_disablecopyonread_encoder_dense_21_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
 *
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
 g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
 
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_encoder_dense_21_bias"/device:CPU:0*
_output_shapes
 Ў
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_encoder_dense_21_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:ќ	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѕ	
value	B	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B з
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *+
dtypes!
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_56Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_57IdentityIdentity_56:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_57Identity_57:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
 '
ё
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934064
input_tensorD
(conv2d_59_conv2d_readvariableop_resource:8
)conv2d_59_biasadd_readvariableop_resource:	=
.batch_normalization_47_readvariableop_resource:	?
0batch_normalization_47_readvariableop_1_resource:	N
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ%batch_normalization_47/AssignNewValueЂ'batch_normalization_47/AssignNewValue_1Ђ6batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_47/ReadVariableOpЂ'batch_normalization_47/ReadVariableOp_1Ђ conv2d_59/BiasAdd/ReadVariableOpЂconv2d_59/Conv2D/ReadVariableOp
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_59/Conv2DConv2Dinput_tensor'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides

 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@m
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@Џ
max_pooling2d_29/MaxPoolMaxPoolconv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides

%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_29/MaxPool:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_47/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ  Ѓ
NoOpNoOp&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ@@: : : : : : 2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ@@
&
_user_specified_nameinput_tensor
Ђ	
з
8__inference_batch_normalization_47_layer_call_fn_1935973

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1933815
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934337
input_tensorD
(conv2d_60_conv2d_readvariableop_resource:8
)conv2d_60_biasadd_readvariableop_resource:	=
.batch_normalization_48_readvariableop_resource:	?
0batch_normalization_48_readvariableop_1_resource:	N
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ6batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_48/ReadVariableOpЂ'batch_normalization_48/ReadVariableOp_1Ђ conv2d_60/BiasAdd/ReadVariableOpЂconv2d_60/Conv2D/ReadVariableOp
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_60/Conv2DConv2Dinput_tensor'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  m
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  Џ
max_pooling2d_30/MaxPoolMaxPoolconv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_30/MaxPool:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_48/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџб
NoOpNoOp7^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ  : : : : : : 2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameinput_tensor
Ш
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_1934164

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
О	

2__inference_encoder_block_29_layer_call_fn_1935558
input_tensor#
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934298x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ@@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:џџџџџџџџџ@@
&
_user_specified_nameinput_tensor

i
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1935947

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ

M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935524
input_tensorC
(conv2d_58_conv2d_readvariableop_resource:8
)conv2d_58_biasadd_readvariableop_resource:	=
.batch_normalization_46_readvariableop_resource:	?
0batch_normalization_46_readvariableop_1_resource:	N
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ6batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_46/ReadVariableOpЂ'batch_normalization_46/ReadVariableOp_1Ђ conv2d_58/BiasAdd/ReadVariableOpЂconv2d_58/Conv2D/ReadVariableOp
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Ж
conv2d_58/Conv2DConv2Dinput_tensor'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџo
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџЏ
max_pooling2d_28/MaxPoolMaxPoolconv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides

%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_28/MaxPool:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_46/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@@б
NoOpNoOp7^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : 2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
П	

2__inference_encoder_block_28_layer_call_fn_1935472
input_tensor"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934259x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
'
ы
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935756
input_tensorC
(conv2d_61_conv2d_readvariableop_resource:@7
)conv2d_61_biasadd_readvariableop_resource:@<
.batch_normalization_49_readvariableop_resource:@>
0batch_normalization_49_readvariableop_1_resource:@M
?batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@
identityЂ%batch_normalization_49/AssignNewValueЂ'batch_normalization_49/AssignNewValue_1Ђ6batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_49/ReadVariableOpЂ'batch_normalization_49/ReadVariableOp_1Ђ conv2d_61/BiasAdd/ReadVariableOpЂconv2d_61/Conv2D/ReadVariableOp
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Г
conv2d_61/Conv2DConv2Dinput_tensor'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@l
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ў
max_pooling2d_31/MaxPoolMaxPoolconv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0в
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_31/MaxPool:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_49/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@Ѓ
NoOpNoOp&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : 2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
ду
Ј#
D__inference_encoder_layer_call_and_return_conditional_losses_1935307
tensor_inputT
9encoder_block_28_conv2d_58_conv2d_readvariableop_resource:I
:encoder_block_28_conv2d_58_biasadd_readvariableop_resource:	N
?encoder_block_28_batch_normalization_46_readvariableop_resource:	P
Aencoder_block_28_batch_normalization_46_readvariableop_1_resource:	_
Pencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	a
Rencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	U
9encoder_block_29_conv2d_59_conv2d_readvariableop_resource:I
:encoder_block_29_conv2d_59_biasadd_readvariableop_resource:	N
?encoder_block_29_batch_normalization_47_readvariableop_resource:	P
Aencoder_block_29_batch_normalization_47_readvariableop_1_resource:	_
Pencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	a
Rencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	U
9encoder_block_30_conv2d_60_conv2d_readvariableop_resource:I
:encoder_block_30_conv2d_60_biasadd_readvariableop_resource:	N
?encoder_block_30_batch_normalization_48_readvariableop_resource:	P
Aencoder_block_30_batch_normalization_48_readvariableop_1_resource:	_
Pencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	a
Rencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	T
9encoder_block_31_conv2d_61_conv2d_readvariableop_resource:@H
:encoder_block_31_conv2d_61_biasadd_readvariableop_resource:@M
?encoder_block_31_batch_normalization_49_readvariableop_resource:@O
Aencoder_block_31_batch_normalization_49_readvariableop_1_resource:@^
Pencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@`
Rencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@;
'dense_20_matmul_readvariableop_resource:
 7
(dense_20_biasadd_readvariableop_resource:	;
'dense_21_matmul_readvariableop_resource:
 7
(dense_21_biasadd_readvariableop_resource:	
identity

identity_1

identity_2Ђdense_20/BiasAdd/ReadVariableOpЂdense_20/MatMul/ReadVariableOpЂdense_21/BiasAdd/ReadVariableOpЂdense_21/MatMul/ReadVariableOpЂ6encoder_block_28/batch_normalization_46/AssignNewValueЂ8encoder_block_28/batch_normalization_46/AssignNewValue_1ЂGencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂIencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_28/batch_normalization_46/ReadVariableOpЂ8encoder_block_28/batch_normalization_46/ReadVariableOp_1Ђ1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOpЂ0encoder_block_28/conv2d_58/Conv2D/ReadVariableOpЂ6encoder_block_29/batch_normalization_47/AssignNewValueЂ8encoder_block_29/batch_normalization_47/AssignNewValue_1ЂGencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂIencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_29/batch_normalization_47/ReadVariableOpЂ8encoder_block_29/batch_normalization_47/ReadVariableOp_1Ђ1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOpЂ0encoder_block_29/conv2d_59/Conv2D/ReadVariableOpЂ6encoder_block_30/batch_normalization_48/AssignNewValueЂ8encoder_block_30/batch_normalization_48/AssignNewValue_1ЂGencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂIencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_30/batch_normalization_48/ReadVariableOpЂ8encoder_block_30/batch_normalization_48/ReadVariableOp_1Ђ1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOpЂ0encoder_block_30/conv2d_60/Conv2D/ReadVariableOpЂ6encoder_block_31/batch_normalization_49/AssignNewValueЂ8encoder_block_31/batch_normalization_49/AssignNewValue_1ЂGencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂIencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_31/batch_normalization_49/ReadVariableOpЂ8encoder_block_31/batch_normalization_49/ReadVariableOp_1Ђ1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOpЂ0encoder_block_31/conv2d_61/Conv2D/ReadVariableOpГ
0encoder_block_28/conv2d_58/Conv2D/ReadVariableOpReadVariableOp9encoder_block_28_conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0и
!encoder_block_28/conv2d_58/Conv2DConv2Dtensor_input8encoder_block_28/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides
Љ
1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_28_conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0б
"encoder_block_28/conv2d_58/BiasAddBiasAdd*encoder_block_28/conv2d_58/Conv2D:output:09encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ
encoder_block_28/conv2d_58/ReluRelu+encoder_block_28/conv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџб
)encoder_block_28/max_pooling2d_28/MaxPoolMaxPool-encoder_block_28/conv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
Г
6encoder_block_28/batch_normalization_46/ReadVariableOpReadVariableOp?encoder_block_28_batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0З
8encoder_block_28/batch_normalization_46/ReadVariableOp_1ReadVariableOpAencoder_block_28_batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0е
Gencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0й
Iencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Н
8encoder_block_28/batch_normalization_46/FusedBatchNormV3FusedBatchNormV32encoder_block_28/max_pooling2d_28/MaxPool:output:0>encoder_block_28/batch_normalization_46/ReadVariableOp:value:0@encoder_block_28/batch_normalization_46/ReadVariableOp_1:value:0Oencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<ц
6encoder_block_28/batch_normalization_46/AssignNewValueAssignVariableOpPencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resourceEencoder_block_28/batch_normalization_46/FusedBatchNormV3:batch_mean:0H^encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(№
8encoder_block_28/batch_normalization_46/AssignNewValue_1AssignVariableOpRencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_28/batch_normalization_46/FusedBatchNormV3:batch_variance:0J^encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Д
0encoder_block_29/conv2d_59/Conv2D/ReadVariableOpReadVariableOp9encoder_block_29_conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
!encoder_block_29/conv2d_59/Conv2DConv2D<encoder_block_28/batch_normalization_46/FusedBatchNormV3:y:08encoder_block_29/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
Љ
1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_29_conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Я
"encoder_block_29/conv2d_59/BiasAddBiasAdd*encoder_block_29/conv2d_59/Conv2D:output:09encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@
encoder_block_29/conv2d_59/ReluRelu+encoder_block_29/conv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@б
)encoder_block_29/max_pooling2d_29/MaxPoolMaxPool-encoder_block_29/conv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
Г
6encoder_block_29/batch_normalization_47/ReadVariableOpReadVariableOp?encoder_block_29_batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0З
8encoder_block_29/batch_normalization_47/ReadVariableOp_1ReadVariableOpAencoder_block_29_batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0е
Gencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0й
Iencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Н
8encoder_block_29/batch_normalization_47/FusedBatchNormV3FusedBatchNormV32encoder_block_29/max_pooling2d_29/MaxPool:output:0>encoder_block_29/batch_normalization_47/ReadVariableOp:value:0@encoder_block_29/batch_normalization_47/ReadVariableOp_1:value:0Oencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
exponential_avg_factor%
з#<ц
6encoder_block_29/batch_normalization_47/AssignNewValueAssignVariableOpPencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resourceEencoder_block_29/batch_normalization_47/FusedBatchNormV3:batch_mean:0H^encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(№
8encoder_block_29/batch_normalization_47/AssignNewValue_1AssignVariableOpRencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_29/batch_normalization_47/FusedBatchNormV3:batch_variance:0J^encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Д
0encoder_block_30/conv2d_60/Conv2D/ReadVariableOpReadVariableOp9encoder_block_30_conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
!encoder_block_30/conv2d_60/Conv2DConv2D<encoder_block_29/batch_normalization_47/FusedBatchNormV3:y:08encoder_block_30/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides
Љ
1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_30_conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Я
"encoder_block_30/conv2d_60/BiasAddBiasAdd*encoder_block_30/conv2d_60/Conv2D:output:09encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  
encoder_block_30/conv2d_60/ReluRelu+encoder_block_30/conv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  б
)encoder_block_30/max_pooling2d_30/MaxPoolMaxPool-encoder_block_30/conv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Г
6encoder_block_30/batch_normalization_48/ReadVariableOpReadVariableOp?encoder_block_30_batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0З
8encoder_block_30/batch_normalization_48/ReadVariableOp_1ReadVariableOpAencoder_block_30_batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0е
Gencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0й
Iencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Н
8encoder_block_30/batch_normalization_48/FusedBatchNormV3FusedBatchNormV32encoder_block_30/max_pooling2d_30/MaxPool:output:0>encoder_block_30/batch_normalization_48/ReadVariableOp:value:0@encoder_block_30/batch_normalization_48/ReadVariableOp_1:value:0Oencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<ц
6encoder_block_30/batch_normalization_48/AssignNewValueAssignVariableOpPencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resourceEencoder_block_30/batch_normalization_48/FusedBatchNormV3:batch_mean:0H^encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(№
8encoder_block_30/batch_normalization_48/AssignNewValue_1AssignVariableOpRencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_30/batch_normalization_48/FusedBatchNormV3:batch_variance:0J^encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
0encoder_block_31/conv2d_61/Conv2D/ReadVariableOpReadVariableOp9encoder_block_31_conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
!encoder_block_31/conv2d_61/Conv2DConv2D<encoder_block_30/batch_normalization_48/FusedBatchNormV3:y:08encoder_block_31/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Ј
1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_31_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
"encoder_block_31/conv2d_61/BiasAddBiasAdd*encoder_block_31/conv2d_61/Conv2D:output:09encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
encoder_block_31/conv2d_61/ReluRelu+encoder_block_31/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@а
)encoder_block_31/max_pooling2d_31/MaxPoolMaxPool-encoder_block_31/conv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
В
6encoder_block_31/batch_normalization_49/ReadVariableOpReadVariableOp?encoder_block_31_batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8encoder_block_31/batch_normalization_49/ReadVariableOp_1ReadVariableOpAencoder_block_31_batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0д
Gencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0и
Iencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0И
8encoder_block_31/batch_normalization_49/FusedBatchNormV3FusedBatchNormV32encoder_block_31/max_pooling2d_31/MaxPool:output:0>encoder_block_31/batch_normalization_49/ReadVariableOp:value:0@encoder_block_31/batch_normalization_49/ReadVariableOp_1:value:0Oencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<ц
6encoder_block_31/batch_normalization_49/AssignNewValueAssignVariableOpPencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resourceEencoder_block_31/batch_normalization_49/FusedBatchNormV3:batch_mean:0H^encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(№
8encoder_block_31/batch_normalization_49/AssignNewValue_1AssignVariableOpRencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_31/batch_normalization_49/FusedBatchNormV3:batch_variance:0J^encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(`
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
flatten_7/ReshapeReshape<encoder_block_31/batch_normalization_49/FusedBatchNormV3:y:0flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_20/MatMulMatMulflatten_7/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџc
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџc
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџi
sampling_7/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
::эЯh
sampling_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_7/strided_sliceStridedSlicesampling_7/Shape:output:0'sampling_7/strided_slice/stack:output:0)sampling_7/strided_slice/stack_1:output:0)sampling_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
sampling_7/Shape_1Shapedense_20/Relu:activations:0*
T0*
_output_shapes
::эЯj
 sampling_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_7/strided_slice_1StridedSlicesampling_7/Shape_1:output:0)sampling_7/strided_slice_1/stack:output:0+sampling_7/strided_slice_1/stack_1:output:0+sampling_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
sampling_7/random_normal/shapePack!sampling_7/strided_slice:output:0#sampling_7/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_7/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_7/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Э
-sampling_7/random_normal/RandomStandardNormalRandomStandardNormal'sampling_7/random_normal/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2ЛЏУ*
seedБџх)И
sampling_7/random_normal/mulMul6sampling_7/random_normal/RandomStandardNormal:output:0(sampling_7/random_normal/stddev:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sampling_7/random_normalAddV2 sampling_7/random_normal/mul:z:0&sampling_7/random_normal/mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџU
sampling_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sampling_7/mulMulsampling_7/mul/x:output:0dense_21/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ\
sampling_7/ExpExpsampling_7/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ|
sampling_7/mul_1Mulsampling_7/Exp:y:0sampling_7/random_normal:z:0*
T0*(
_output_shapes
:џџџџџџџџџ}
sampling_7/addAddV2dense_20/Relu:activations:0sampling_7/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџk
IdentityIdentitydense_20/Relu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџm

Identity_1Identitydense_21/Relu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџd

Identity_2Identitysampling_7/add:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџр
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp7^encoder_block_28/batch_normalization_46/AssignNewValue9^encoder_block_28/batch_normalization_46/AssignNewValue_1H^encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpJ^encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_17^encoder_block_28/batch_normalization_46/ReadVariableOp9^encoder_block_28/batch_normalization_46/ReadVariableOp_12^encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp1^encoder_block_28/conv2d_58/Conv2D/ReadVariableOp7^encoder_block_29/batch_normalization_47/AssignNewValue9^encoder_block_29/batch_normalization_47/AssignNewValue_1H^encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpJ^encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_17^encoder_block_29/batch_normalization_47/ReadVariableOp9^encoder_block_29/batch_normalization_47/ReadVariableOp_12^encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp1^encoder_block_29/conv2d_59/Conv2D/ReadVariableOp7^encoder_block_30/batch_normalization_48/AssignNewValue9^encoder_block_30/batch_normalization_48/AssignNewValue_1H^encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpJ^encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_17^encoder_block_30/batch_normalization_48/ReadVariableOp9^encoder_block_30/batch_normalization_48/ReadVariableOp_12^encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp1^encoder_block_30/conv2d_60/Conv2D/ReadVariableOp7^encoder_block_31/batch_normalization_49/AssignNewValue9^encoder_block_31/batch_normalization_49/AssignNewValue_1H^encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpJ^encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_17^encoder_block_31/batch_normalization_49/ReadVariableOp9^encoder_block_31/batch_normalization_49/ReadVariableOp_12^encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp1^encoder_block_31/conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2t
8encoder_block_28/batch_normalization_46/AssignNewValue_18encoder_block_28/batch_normalization_46/AssignNewValue_12p
6encoder_block_28/batch_normalization_46/AssignNewValue6encoder_block_28/batch_normalization_46/AssignNewValue2
Iencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpGencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_28/batch_normalization_46/ReadVariableOp_18encoder_block_28/batch_normalization_46/ReadVariableOp_12p
6encoder_block_28/batch_normalization_46/ReadVariableOp6encoder_block_28/batch_normalization_46/ReadVariableOp2f
1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp2d
0encoder_block_28/conv2d_58/Conv2D/ReadVariableOp0encoder_block_28/conv2d_58/Conv2D/ReadVariableOp2t
8encoder_block_29/batch_normalization_47/AssignNewValue_18encoder_block_29/batch_normalization_47/AssignNewValue_12p
6encoder_block_29/batch_normalization_47/AssignNewValue6encoder_block_29/batch_normalization_47/AssignNewValue2
Iencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpGencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_29/batch_normalization_47/ReadVariableOp_18encoder_block_29/batch_normalization_47/ReadVariableOp_12p
6encoder_block_29/batch_normalization_47/ReadVariableOp6encoder_block_29/batch_normalization_47/ReadVariableOp2f
1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp2d
0encoder_block_29/conv2d_59/Conv2D/ReadVariableOp0encoder_block_29/conv2d_59/Conv2D/ReadVariableOp2t
8encoder_block_30/batch_normalization_48/AssignNewValue_18encoder_block_30/batch_normalization_48/AssignNewValue_12p
6encoder_block_30/batch_normalization_48/AssignNewValue6encoder_block_30/batch_normalization_48/AssignNewValue2
Iencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpGencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_30/batch_normalization_48/ReadVariableOp_18encoder_block_30/batch_normalization_48/ReadVariableOp_12p
6encoder_block_30/batch_normalization_48/ReadVariableOp6encoder_block_30/batch_normalization_48/ReadVariableOp2f
1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp2d
0encoder_block_30/conv2d_60/Conv2D/ReadVariableOp0encoder_block_30/conv2d_60/Conv2D/ReadVariableOp2t
8encoder_block_31/batch_normalization_49/AssignNewValue_18encoder_block_31/batch_normalization_49/AssignNewValue_12p
6encoder_block_31/batch_normalization_49/AssignNewValue6encoder_block_31/batch_normalization_49/AssignNewValue2
Iencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpGencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_31/batch_normalization_49/ReadVariableOp_18encoder_block_31/batch_normalization_49/ReadVariableOp_12p
6encoder_block_31/batch_normalization_49/ReadVariableOp6encoder_block_31/batch_normalization_49/ReadVariableOp2f
1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp2d
0encoder_block_31/conv2d_61/Conv2D/ReadVariableOp0encoder_block_31/conv2d_61/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nametensor_input
о
Ђ
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1933739

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
М	

2__inference_encoder_block_29_layer_call_fn_1935541
input_tensor#
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934064x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ@@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:џџџџџџџџџ@@
&
_user_specified_nameinput_tensor
о
Ђ
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1933815

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
G
+__inference_flatten_7_layer_call_fn_1935787

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1934164a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
І'
№
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934024
input_tensorC
(conv2d_58_conv2d_readvariableop_resource:8
)conv2d_58_biasadd_readvariableop_resource:	=
.batch_normalization_46_readvariableop_resource:	?
0batch_normalization_46_readvariableop_1_resource:	N
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ%batch_normalization_46/AssignNewValueЂ'batch_normalization_46/AssignNewValue_1Ђ6batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_46/ReadVariableOpЂ'batch_normalization_46/ReadVariableOp_1Ђ conv2d_58/BiasAdd/ReadVariableOpЂconv2d_58/Conv2D/ReadVariableOp
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Ж
conv2d_58/Conv2DConv2Dinput_tensor'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџo
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџЏ
max_pooling2d_28/MaxPoolMaxPoolconv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides

%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_28/MaxPool:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_46/AssignNewValueAssignVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource4batch_normalization_46/FusedBatchNormV3:batch_mean:07^batch_normalization_46/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_46/AssignNewValue_1AssignVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_46/FusedBatchNormV3:batch_variance:09^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_46/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@@Ѓ
NoOpNoOp&^batch_normalization_46/AssignNewValue(^batch_normalization_46/AssignNewValue_17^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : 2R
'batch_normalization_46/AssignNewValue_1'batch_normalization_46/AssignNewValue_12N
%batch_normalization_46/AssignNewValue%batch_normalization_46/AssignNewValue2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
 '
ё
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935584
input_tensorD
(conv2d_59_conv2d_readvariableop_resource:8
)conv2d_59_biasadd_readvariableop_resource:	=
.batch_normalization_47_readvariableop_resource:	?
0batch_normalization_47_readvariableop_1_resource:	N
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ%batch_normalization_47/AssignNewValueЂ'batch_normalization_47/AssignNewValue_1Ђ6batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_47/ReadVariableOpЂ'batch_normalization_47/ReadVariableOp_1Ђ conv2d_59/BiasAdd/ReadVariableOpЂconv2d_59/Conv2D/ReadVariableOp
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_59/Conv2DConv2Dinput_tensor'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides

 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@m
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@Џ
max_pooling2d_29/MaxPoolMaxPoolconv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides

%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_29/MaxPool:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_47/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ  Ѓ
NoOpNoOp&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ@@: : : : : : 2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ@@
&
_user_specified_nameinput_tensor
В
v
G__inference_sampling_7_layer_call_and_return_conditional_losses_1935865
inputs_0
inputs_1
identityK
ShapeShapeinputs_0*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_0*
T0*
_output_shapes
::эЯ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2кБ*
seedБџх)
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:џџџџџџџџџ}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulmul/x:output:0inputs_1*
T0*(
_output_shapes
:џџџџџџџџџF
ExpExpmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:џџџџџџџџџT
addAddV2inputs_0	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
б
Ё
)__inference_encoder_layer_call_fn_1934541
input_1"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
 

unknown_24:	

unknown_25:
 

unknown_26:	
identity

identity_1

identity_2ЂStatefulPartitionedCallх
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1934478p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
р
І
)__inference_encoder_layer_call_fn_1935111
tensor_input"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
 

unknown_24:	

unknown_25:
 

unknown_26:	
identity

identity_1

identity_2ЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCalltensor_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1934478p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nametensor_input

i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1933696

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Ђ
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1933891

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
'
ы
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934144
input_tensorC
(conv2d_61_conv2d_readvariableop_resource:@7
)conv2d_61_biasadd_readvariableop_resource:@<
.batch_normalization_49_readvariableop_resource:@>
0batch_normalization_49_readvariableop_1_resource:@M
?batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@
identityЂ%batch_normalization_49/AssignNewValueЂ'batch_normalization_49/AssignNewValue_1Ђ6batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_49/ReadVariableOpЂ'batch_normalization_49/ReadVariableOp_1Ђ conv2d_61/BiasAdd/ReadVariableOpЂconv2d_61/Conv2D/ReadVariableOp
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Г
conv2d_61/Conv2DConv2Dinput_tensor'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@l
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ў
max_pooling2d_31/MaxPoolMaxPoolconv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0в
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_31/MaxPool:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_49/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@Ѓ
NoOpNoOp&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : 2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor

Ц
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1933797

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
С
N
2__inference_max_pooling2d_29_layer_call_fn_1935942

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1933772
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936135

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_1933848

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934298
input_tensorD
(conv2d_59_conv2d_readvariableop_resource:8
)conv2d_59_biasadd_readvariableop_resource:	=
.batch_normalization_47_readvariableop_resource:	?
0batch_normalization_47_readvariableop_1_resource:	N
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ6batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_47/ReadVariableOpЂ'batch_normalization_47/ReadVariableOp_1Ђ conv2d_59/BiasAdd/ReadVariableOpЂconv2d_59/Conv2D/ReadVariableOp
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_59/Conv2DConv2Dinput_tensor'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides

 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@m
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@Џ
max_pooling2d_29/MaxPoolMaxPoolconv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides

%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_29/MaxPool:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_47/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ  б
NoOpNoOp7^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ@@: : : : : : 2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ@@
&
_user_specified_nameinput_tensor
О	

2__inference_encoder_block_30_layer_call_fn_1935644
input_tensor#
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identityЂStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934337x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ  : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameinput_tensor

Ц
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936063

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Т
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1933949

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ж
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@д
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
 7
Ў
D__inference_encoder_layer_call_and_return_conditional_losses_1934405
input_13
encoder_block_28_1934260:'
encoder_block_28_1934262:	'
encoder_block_28_1934264:	'
encoder_block_28_1934266:	'
encoder_block_28_1934268:	'
encoder_block_28_1934270:	4
encoder_block_29_1934299:'
encoder_block_29_1934301:	'
encoder_block_29_1934303:	'
encoder_block_29_1934305:	'
encoder_block_29_1934307:	'
encoder_block_29_1934309:	4
encoder_block_30_1934338:'
encoder_block_30_1934340:	'
encoder_block_30_1934342:	'
encoder_block_30_1934344:	'
encoder_block_30_1934346:	'
encoder_block_30_1934348:	3
encoder_block_31_1934377:@&
encoder_block_31_1934379:@&
encoder_block_31_1934381:@&
encoder_block_31_1934383:@&
encoder_block_31_1934385:@&
encoder_block_31_1934387:@$
dense_20_1934391:
 
dense_20_1934393:	$
dense_21_1934396:
 
dense_21_1934398:	
identity

identity_1

identity_2Ђ dense_20/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ(encoder_block_28/StatefulPartitionedCallЂ(encoder_block_29/StatefulPartitionedCallЂ(encoder_block_30/StatefulPartitionedCallЂ(encoder_block_31/StatefulPartitionedCallЂ"sampling_7/StatefulPartitionedCall
(encoder_block_28/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_block_28_1934260encoder_block_28_1934262encoder_block_28_1934264encoder_block_28_1934266encoder_block_28_1934268encoder_block_28_1934270*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934259М
(encoder_block_29/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_28/StatefulPartitionedCall:output:0encoder_block_29_1934299encoder_block_29_1934301encoder_block_29_1934303encoder_block_29_1934305encoder_block_29_1934307encoder_block_29_1934309*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934298М
(encoder_block_30/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_29/StatefulPartitionedCall:output:0encoder_block_30_1934338encoder_block_30_1934340encoder_block_30_1934342encoder_block_30_1934344encoder_block_30_1934346encoder_block_30_1934348*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934337Л
(encoder_block_31/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_30/StatefulPartitionedCall:output:0encoder_block_31_1934377encoder_block_31_1934379encoder_block_31_1934381encoder_block_31_1934383encoder_block_31_1934385encoder_block_31_1934387*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934376ь
flatten_7/PartitionedCallPartitionedCall1encoder_block_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1934164
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_20_1934391dense_20_1934393*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1934177
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_1934396dense_21_1934398*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1934194Ђ
"sampling_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_sampling_7_layer_call_and_return_conditional_losses_1934226y
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ{

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ}

Identity_2Identity+sampling_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџн
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_28/StatefulPartitionedCall)^encoder_block_29/StatefulPartitionedCall)^encoder_block_30/StatefulPartitionedCall)^encoder_block_31/StatefulPartitionedCall#^sampling_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_28/StatefulPartitionedCall(encoder_block_28/StatefulPartitionedCall2T
(encoder_block_29/StatefulPartitionedCall(encoder_block_29/StatefulPartitionedCall2T
(encoder_block_30/StatefulPartitionedCall(encoder_block_30/StatefulPartitionedCall2T
(encoder_block_31/StatefulPartitionedCall(encoder_block_31/StatefulPartitionedCall2H
"sampling_7/StatefulPartitionedCall"sampling_7/StatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
а

*__inference_dense_21_layer_call_fn_1935822

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1934194p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
 	
з
8__inference_batch_normalization_47_layer_call_fn_1935960

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1933797
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§б
ё"
"__inference__wrapped_model_1933690
input_1\
Aencoder_encoder_block_28_conv2d_58_conv2d_readvariableop_resource:Q
Bencoder_encoder_block_28_conv2d_58_biasadd_readvariableop_resource:	V
Gencoder_encoder_block_28_batch_normalization_46_readvariableop_resource:	X
Iencoder_encoder_block_28_batch_normalization_46_readvariableop_1_resource:	g
Xencoder_encoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	i
Zencoder_encoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	]
Aencoder_encoder_block_29_conv2d_59_conv2d_readvariableop_resource:Q
Bencoder_encoder_block_29_conv2d_59_biasadd_readvariableop_resource:	V
Gencoder_encoder_block_29_batch_normalization_47_readvariableop_resource:	X
Iencoder_encoder_block_29_batch_normalization_47_readvariableop_1_resource:	g
Xencoder_encoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	i
Zencoder_encoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	]
Aencoder_encoder_block_30_conv2d_60_conv2d_readvariableop_resource:Q
Bencoder_encoder_block_30_conv2d_60_biasadd_readvariableop_resource:	V
Gencoder_encoder_block_30_batch_normalization_48_readvariableop_resource:	X
Iencoder_encoder_block_30_batch_normalization_48_readvariableop_1_resource:	g
Xencoder_encoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	i
Zencoder_encoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	\
Aencoder_encoder_block_31_conv2d_61_conv2d_readvariableop_resource:@P
Bencoder_encoder_block_31_conv2d_61_biasadd_readvariableop_resource:@U
Gencoder_encoder_block_31_batch_normalization_49_readvariableop_resource:@W
Iencoder_encoder_block_31_batch_normalization_49_readvariableop_1_resource:@f
Xencoder_encoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@h
Zencoder_encoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@C
/encoder_dense_20_matmul_readvariableop_resource:
 ?
0encoder_dense_20_biasadd_readvariableop_resource:	C
/encoder_dense_21_matmul_readvariableop_resource:
 ?
0encoder_dense_21_biasadd_readvariableop_resource:	
identity

identity_1

identity_2Ђ'encoder/dense_20/BiasAdd/ReadVariableOpЂ&encoder/dense_20/MatMul/ReadVariableOpЂ'encoder/dense_21/BiasAdd/ReadVariableOpЂ&encoder/dense_21/MatMul/ReadVariableOpЂOencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂQencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ>encoder/encoder_block_28/batch_normalization_46/ReadVariableOpЂ@encoder/encoder_block_28/batch_normalization_46/ReadVariableOp_1Ђ9encoder/encoder_block_28/conv2d_58/BiasAdd/ReadVariableOpЂ8encoder/encoder_block_28/conv2d_58/Conv2D/ReadVariableOpЂOencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂQencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ>encoder/encoder_block_29/batch_normalization_47/ReadVariableOpЂ@encoder/encoder_block_29/batch_normalization_47/ReadVariableOp_1Ђ9encoder/encoder_block_29/conv2d_59/BiasAdd/ReadVariableOpЂ8encoder/encoder_block_29/conv2d_59/Conv2D/ReadVariableOpЂOencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂQencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ>encoder/encoder_block_30/batch_normalization_48/ReadVariableOpЂ@encoder/encoder_block_30/batch_normalization_48/ReadVariableOp_1Ђ9encoder/encoder_block_30/conv2d_60/BiasAdd/ReadVariableOpЂ8encoder/encoder_block_30/conv2d_60/Conv2D/ReadVariableOpЂOencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂQencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ>encoder/encoder_block_31/batch_normalization_49/ReadVariableOpЂ@encoder/encoder_block_31/batch_normalization_49/ReadVariableOp_1Ђ9encoder/encoder_block_31/conv2d_61/BiasAdd/ReadVariableOpЂ8encoder/encoder_block_31/conv2d_61/Conv2D/ReadVariableOpУ
8encoder/encoder_block_28/conv2d_58/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_28_conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0у
)encoder/encoder_block_28/conv2d_58/Conv2DConv2Dinput_1@encoder/encoder_block_28/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides
Й
9encoder/encoder_block_28/conv2d_58/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_28_conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0щ
*encoder/encoder_block_28/conv2d_58/BiasAddBiasAdd2encoder/encoder_block_28/conv2d_58/Conv2D:output:0Aencoder/encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџЁ
'encoder/encoder_block_28/conv2d_58/ReluRelu3encoder/encoder_block_28/conv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџс
1encoder/encoder_block_28/max_pooling2d_28/MaxPoolMaxPool5encoder/encoder_block_28/conv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
У
>encoder/encoder_block_28/batch_normalization_46/ReadVariableOpReadVariableOpGencoder_encoder_block_28_batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@encoder/encoder_block_28/batch_normalization_46/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_28_batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0х
Oencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0щ
Qencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0п
@encoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_28/max_pooling2d_28/MaxPool:output:0Fencoder/encoder_block_28/batch_normalization_46/ReadVariableOp:value:0Hencoder/encoder_block_28/batch_normalization_46/ReadVariableOp_1:value:0Wencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( Ф
8encoder/encoder_block_29/conv2d_59/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_29_conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)encoder/encoder_block_29/conv2d_59/Conv2DConv2DDencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3:y:0@encoder/encoder_block_29/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
Й
9encoder/encoder_block_29/conv2d_59/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_29_conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ч
*encoder/encoder_block_29/conv2d_59/BiasAddBiasAdd2encoder/encoder_block_29/conv2d_59/Conv2D:output:0Aencoder/encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@
'encoder/encoder_block_29/conv2d_59/ReluRelu3encoder/encoder_block_29/conv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@с
1encoder/encoder_block_29/max_pooling2d_29/MaxPoolMaxPool5encoder/encoder_block_29/conv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
У
>encoder/encoder_block_29/batch_normalization_47/ReadVariableOpReadVariableOpGencoder_encoder_block_29_batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@encoder/encoder_block_29/batch_normalization_47/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_29_batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0х
Oencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0щ
Qencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0п
@encoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_29/max_pooling2d_29/MaxPool:output:0Fencoder/encoder_block_29/batch_normalization_47/ReadVariableOp:value:0Hencoder/encoder_block_29/batch_normalization_47/ReadVariableOp_1:value:0Wencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
is_training( Ф
8encoder/encoder_block_30/conv2d_60/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_30_conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
)encoder/encoder_block_30/conv2d_60/Conv2DConv2DDencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3:y:0@encoder/encoder_block_30/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides
Й
9encoder/encoder_block_30/conv2d_60/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_30_conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ч
*encoder/encoder_block_30/conv2d_60/BiasAddBiasAdd2encoder/encoder_block_30/conv2d_60/Conv2D:output:0Aencoder/encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  
'encoder/encoder_block_30/conv2d_60/ReluRelu3encoder/encoder_block_30/conv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  с
1encoder/encoder_block_30/max_pooling2d_30/MaxPoolMaxPool5encoder/encoder_block_30/conv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
У
>encoder/encoder_block_30/batch_normalization_48/ReadVariableOpReadVariableOpGencoder_encoder_block_30_batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0Ч
@encoder/encoder_block_30/batch_normalization_48/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_30_batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0х
Oencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0щ
Qencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0п
@encoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_30/max_pooling2d_30/MaxPool:output:0Fencoder/encoder_block_30/batch_normalization_48/ReadVariableOp:value:0Hencoder/encoder_block_30/batch_normalization_48/ReadVariableOp_1:value:0Wencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( У
8encoder/encoder_block_31/conv2d_61/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_31_conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
)encoder/encoder_block_31/conv2d_61/Conv2DConv2DDencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3:y:0@encoder/encoder_block_31/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
И
9encoder/encoder_block_31/conv2d_61/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_31_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ц
*encoder/encoder_block_31/conv2d_61/BiasAddBiasAdd2encoder/encoder_block_31/conv2d_61/Conv2D:output:0Aencoder/encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
'encoder/encoder_block_31/conv2d_61/ReluRelu3encoder/encoder_block_31/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@р
1encoder/encoder_block_31/max_pooling2d_31/MaxPoolMaxPool5encoder/encoder_block_31/conv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
Т
>encoder/encoder_block_31/batch_normalization_49/ReadVariableOpReadVariableOpGencoder_encoder_block_31_batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0Ц
@encoder/encoder_block_31/batch_normalization_49/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_31_batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0ф
Oencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ш
Qencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
@encoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_31/max_pooling2d_31/MaxPool:output:0Fencoder/encoder_block_31/batch_normalization_49/ReadVariableOp:value:0Hencoder/encoder_block_31/batch_normalization_49/ReadVariableOp_1:value:0Wencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( h
encoder/flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   П
encoder/flatten_7/ReshapeReshapeDencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3:y:0 encoder/flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
&encoder/dense_20/MatMul/ReadVariableOpReadVariableOp/encoder_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0Ј
encoder/dense_20/MatMulMatMul"encoder/flatten_7/Reshape:output:0.encoder/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
'encoder/dense_20/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Њ
encoder/dense_20/BiasAddBiasAdd!encoder/dense_20/MatMul:product:0/encoder/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
encoder/dense_20/ReluRelu!encoder/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&encoder/dense_21/MatMul/ReadVariableOpReadVariableOp/encoder_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0Ј
encoder/dense_21/MatMulMatMul"encoder/flatten_7/Reshape:output:0.encoder/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
'encoder/dense_21/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Њ
encoder/dense_21/BiasAddBiasAdd!encoder/dense_21/MatMul:product:0/encoder/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
encoder/dense_21/ReluRelu!encoder/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџy
encoder/sampling_7/ShapeShape#encoder/dense_20/Relu:activations:0*
T0*
_output_shapes
::эЯp
&encoder/sampling_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(encoder/sampling_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 encoder/sampling_7/strided_sliceStridedSlice!encoder/sampling_7/Shape:output:0/encoder/sampling_7/strided_slice/stack:output:01encoder/sampling_7/strided_slice/stack_1:output:01encoder/sampling_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
encoder/sampling_7/Shape_1Shape#encoder/dense_20/Relu:activations:0*
T0*
_output_shapes
::эЯr
(encoder/sampling_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"encoder/sampling_7/strided_slice_1StridedSlice#encoder/sampling_7/Shape_1:output:01encoder/sampling_7/strided_slice_1/stack:output:03encoder/sampling_7/strided_slice_1/stack_1:output:03encoder/sampling_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
&encoder/sampling_7/random_normal/shapePack)encoder/sampling_7/strided_slice:output:0+encoder/sampling_7/strided_slice_1:output:0*
N*
T0*
_output_shapes
:j
%encoder/sampling_7/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'encoder/sampling_7/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?н
5encoder/sampling_7/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_7/random_normal/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2З*
seedБџх)а
$encoder/sampling_7/random_normal/mulMul>encoder/sampling_7/random_normal/RandomStandardNormal:output:00encoder/sampling_7/random_normal/stddev:output:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
 encoder/sampling_7/random_normalAddV2(encoder/sampling_7/random_normal/mul:z:0.encoder/sampling_7/random_normal/mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ]
encoder/sampling_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
encoder/sampling_7/mulMul!encoder/sampling_7/mul/x:output:0#encoder/dense_21/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџl
encoder/sampling_7/ExpExpencoder/sampling_7/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
encoder/sampling_7/mul_1Mulencoder/sampling_7/Exp:y:0$encoder/sampling_7/random_normal:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
encoder/sampling_7/addAddV2#encoder/dense_20/Relu:activations:0encoder/sampling_7/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџs
IdentityIdentity#encoder/dense_20/Relu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџu

Identity_1Identity#encoder/dense_21/Relu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџl

Identity_2Identityencoder/sampling_7/add:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ№
NoOpNoOp(^encoder/dense_20/BiasAdd/ReadVariableOp'^encoder/dense_20/MatMul/ReadVariableOp(^encoder/dense_21/BiasAdd/ReadVariableOp'^encoder/dense_21/MatMul/ReadVariableOpP^encoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_28/batch_normalization_46/ReadVariableOpA^encoder/encoder_block_28/batch_normalization_46/ReadVariableOp_1:^encoder/encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp9^encoder/encoder_block_28/conv2d_58/Conv2D/ReadVariableOpP^encoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_29/batch_normalization_47/ReadVariableOpA^encoder/encoder_block_29/batch_normalization_47/ReadVariableOp_1:^encoder/encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp9^encoder/encoder_block_29/conv2d_59/Conv2D/ReadVariableOpP^encoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_30/batch_normalization_48/ReadVariableOpA^encoder/encoder_block_30/batch_normalization_48/ReadVariableOp_1:^encoder/encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp9^encoder/encoder_block_30/conv2d_60/Conv2D/ReadVariableOpP^encoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_31/batch_normalization_49/ReadVariableOpA^encoder/encoder_block_31/batch_normalization_49/ReadVariableOp_1:^encoder/encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp9^encoder/encoder_block_31/conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'encoder/dense_20/BiasAdd/ReadVariableOp'encoder/dense_20/BiasAdd/ReadVariableOp2P
&encoder/dense_20/MatMul/ReadVariableOp&encoder/dense_20/MatMul/ReadVariableOp2R
'encoder/dense_21/BiasAdd/ReadVariableOp'encoder/dense_21/BiasAdd/ReadVariableOp2P
&encoder/dense_21/MatMul/ReadVariableOp&encoder/dense_21/MatMul/ReadVariableOp2І
Qencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12Ђ
Oencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp2
@encoder/encoder_block_28/batch_normalization_46/ReadVariableOp_1@encoder/encoder_block_28/batch_normalization_46/ReadVariableOp_12
>encoder/encoder_block_28/batch_normalization_46/ReadVariableOp>encoder/encoder_block_28/batch_normalization_46/ReadVariableOp2v
9encoder/encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp9encoder/encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_28/conv2d_58/Conv2D/ReadVariableOp8encoder/encoder_block_28/conv2d_58/Conv2D/ReadVariableOp2І
Qencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12Ђ
Oencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2
@encoder/encoder_block_29/batch_normalization_47/ReadVariableOp_1@encoder/encoder_block_29/batch_normalization_47/ReadVariableOp_12
>encoder/encoder_block_29/batch_normalization_47/ReadVariableOp>encoder/encoder_block_29/batch_normalization_47/ReadVariableOp2v
9encoder/encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp9encoder/encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_29/conv2d_59/Conv2D/ReadVariableOp8encoder/encoder_block_29/conv2d_59/Conv2D/ReadVariableOp2І
Qencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12Ђ
Oencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2
@encoder/encoder_block_30/batch_normalization_48/ReadVariableOp_1@encoder/encoder_block_30/batch_normalization_48/ReadVariableOp_12
>encoder/encoder_block_30/batch_normalization_48/ReadVariableOp>encoder/encoder_block_30/batch_normalization_48/ReadVariableOp2v
9encoder/encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp9encoder/encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_30/conv2d_60/Conv2D/ReadVariableOp8encoder/encoder_block_30/conv2d_60/Conv2D/ReadVariableOp2І
Qencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12Ђ
Oencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2
@encoder/encoder_block_31/batch_normalization_49/ReadVariableOp_1@encoder/encoder_block_31/batch_normalization_49/ReadVariableOp_12
>encoder/encoder_block_31/batch_normalization_49/ReadVariableOp>encoder/encoder_block_31/batch_normalization_49/ReadVariableOp2v
9encoder/encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp9encoder/encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_31/conv2d_61/Conv2D/ReadVariableOp8encoder/encoder_block_31/conv2d_61/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
	
г
8__inference_batch_normalization_49_layer_call_fn_1936117

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1933967
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
І
t
G__inference_sampling_7_layer_call_and_return_conditional_losses_1934226

inputs
inputs_1
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
Shape_1Shapeinputs*
T0*
_output_shapes
::эЯ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?З
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2М*
seedБџх)
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:џџџџџџџџџ}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulmul/x:output:0inputs_1*
T0*(
_output_shapes
:џџџџџџџџџF
ExpExpmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:џџџџџџџџџR
addAddV2inputs	mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 	
з
8__inference_batch_normalization_46_layer_call_fn_1935888

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1933721
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1933873

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1935875

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

љ
E__inference_dense_21_layer_call_and_return_conditional_losses_1935833

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ќ

M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934259
input_tensorC
(conv2d_58_conv2d_readvariableop_resource:8
)conv2d_58_biasadd_readvariableop_resource:	=
.batch_normalization_46_readvariableop_resource:	?
0batch_normalization_46_readvariableop_1_resource:	N
?batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ6batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_46/ReadVariableOpЂ'batch_normalization_46/ReadVariableOp_1Ђ conv2d_58/BiasAdd/ReadVariableOpЂconv2d_58/Conv2D/ReadVariableOp
conv2d_58/Conv2D/ReadVariableOpReadVariableOp(conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Ж
conv2d_58/Conv2DConv2Dinput_tensor'conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides

 conv2d_58/BiasAdd/ReadVariableOpReadVariableOp)conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_58/BiasAddBiasAddconv2d_58/Conv2D:output:0(conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџo
conv2d_58/ReluReluconv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџЏ
max_pooling2d_28/MaxPoolMaxPoolconv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides

%batch_normalization_46/ReadVariableOpReadVariableOp.batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_46/ReadVariableOp_1ReadVariableOp0batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_46/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_28/MaxPool:output:0-batch_normalization_46/ReadVariableOp:value:0/batch_normalization_46/ReadVariableOp_1:value:0>batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_46/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@@б
NoOpNoOp7^batch_normalization_46/FusedBatchNormV3/ReadVariableOp9^batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_46/ReadVariableOp(^batch_normalization_46/ReadVariableOp_1!^conv2d_58/BiasAdd/ReadVariableOp ^conv2d_58/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : 2t
8batch_normalization_46/FusedBatchNormV3/ReadVariableOp_18batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_46/FusedBatchNormV3/ReadVariableOp6batch_normalization_46/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_46/ReadVariableOp_1'batch_normalization_46/ReadVariableOp_12N
%batch_normalization_46/ReadVariableOp%batch_normalization_46/ReadVariableOp2D
 conv2d_58/BiasAdd/ReadVariableOp conv2d_58/BiasAdd/ReadVariableOp2B
conv2d_58/Conv2D/ReadVariableOpconv2d_58/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
ьН
и
D__inference_encoder_layer_call_and_return_conditional_losses_1935438
tensor_inputT
9encoder_block_28_conv2d_58_conv2d_readvariableop_resource:I
:encoder_block_28_conv2d_58_biasadd_readvariableop_resource:	N
?encoder_block_28_batch_normalization_46_readvariableop_resource:	P
Aencoder_block_28_batch_normalization_46_readvariableop_1_resource:	_
Pencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resource:	a
Rencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource:	U
9encoder_block_29_conv2d_59_conv2d_readvariableop_resource:I
:encoder_block_29_conv2d_59_biasadd_readvariableop_resource:	N
?encoder_block_29_batch_normalization_47_readvariableop_resource:	P
Aencoder_block_29_batch_normalization_47_readvariableop_1_resource:	_
Pencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	a
Rencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	U
9encoder_block_30_conv2d_60_conv2d_readvariableop_resource:I
:encoder_block_30_conv2d_60_biasadd_readvariableop_resource:	N
?encoder_block_30_batch_normalization_48_readvariableop_resource:	P
Aencoder_block_30_batch_normalization_48_readvariableop_1_resource:	_
Pencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	a
Rencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	T
9encoder_block_31_conv2d_61_conv2d_readvariableop_resource:@H
:encoder_block_31_conv2d_61_biasadd_readvariableop_resource:@M
?encoder_block_31_batch_normalization_49_readvariableop_resource:@O
Aencoder_block_31_batch_normalization_49_readvariableop_1_resource:@^
Pencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@`
Rencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@;
'dense_20_matmul_readvariableop_resource:
 7
(dense_20_biasadd_readvariableop_resource:	;
'dense_21_matmul_readvariableop_resource:
 7
(dense_21_biasadd_readvariableop_resource:	
identity

identity_1

identity_2Ђdense_20/BiasAdd/ReadVariableOpЂdense_20/MatMul/ReadVariableOpЂdense_21/BiasAdd/ReadVariableOpЂdense_21/MatMul/ReadVariableOpЂGencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpЂIencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_28/batch_normalization_46/ReadVariableOpЂ8encoder_block_28/batch_normalization_46/ReadVariableOp_1Ђ1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOpЂ0encoder_block_28/conv2d_58/Conv2D/ReadVariableOpЂGencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂIencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_29/batch_normalization_47/ReadVariableOpЂ8encoder_block_29/batch_normalization_47/ReadVariableOp_1Ђ1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOpЂ0encoder_block_29/conv2d_59/Conv2D/ReadVariableOpЂGencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂIencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_30/batch_normalization_48/ReadVariableOpЂ8encoder_block_30/batch_normalization_48/ReadVariableOp_1Ђ1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOpЂ0encoder_block_30/conv2d_60/Conv2D/ReadVariableOpЂGencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂIencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ6encoder_block_31/batch_normalization_49/ReadVariableOpЂ8encoder_block_31/batch_normalization_49/ReadVariableOp_1Ђ1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOpЂ0encoder_block_31/conv2d_61/Conv2D/ReadVariableOpГ
0encoder_block_28/conv2d_58/Conv2D/ReadVariableOpReadVariableOp9encoder_block_28_conv2d_58_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0и
!encoder_block_28/conv2d_58/Conv2DConv2Dtensor_input8encoder_block_28/conv2d_58/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ*
paddingSAME*
strides
Љ
1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_28_conv2d_58_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0б
"encoder_block_28/conv2d_58/BiasAddBiasAdd*encoder_block_28/conv2d_58/Conv2D:output:09encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:џџџџџџџџџ
encoder_block_28/conv2d_58/ReluRelu+encoder_block_28/conv2d_58/BiasAdd:output:0*
T0*2
_output_shapes 
:џџџџџџџџџб
)encoder_block_28/max_pooling2d_28/MaxPoolMaxPool-encoder_block_28/conv2d_58/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ@@*
ksize
*
paddingVALID*
strides
Г
6encoder_block_28/batch_normalization_46/ReadVariableOpReadVariableOp?encoder_block_28_batch_normalization_46_readvariableop_resource*
_output_shapes	
:*
dtype0З
8encoder_block_28/batch_normalization_46/ReadVariableOp_1ReadVariableOpAencoder_block_28_batch_normalization_46_readvariableop_1_resource*
_output_shapes	
:*
dtype0е
Gencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0й
Iencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_28_batch_normalization_46_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Џ
8encoder_block_28/batch_normalization_46/FusedBatchNormV3FusedBatchNormV32encoder_block_28/max_pooling2d_28/MaxPool:output:0>encoder_block_28/batch_normalization_46/ReadVariableOp:value:0@encoder_block_28/batch_normalization_46/ReadVariableOp_1:value:0Oencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ@@:::::*
epsilon%o:*
is_training( Д
0encoder_block_29/conv2d_59/Conv2D/ReadVariableOpReadVariableOp9encoder_block_29_conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
!encoder_block_29/conv2d_59/Conv2DConv2D<encoder_block_28/batch_normalization_46/FusedBatchNormV3:y:08encoder_block_29/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides
Љ
1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_29_conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Я
"encoder_block_29/conv2d_59/BiasAddBiasAdd*encoder_block_29/conv2d_59/Conv2D:output:09encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@
encoder_block_29/conv2d_59/ReluRelu+encoder_block_29/conv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@б
)encoder_block_29/max_pooling2d_29/MaxPoolMaxPool-encoder_block_29/conv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides
Г
6encoder_block_29/batch_normalization_47/ReadVariableOpReadVariableOp?encoder_block_29_batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0З
8encoder_block_29/batch_normalization_47/ReadVariableOp_1ReadVariableOpAencoder_block_29_batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0е
Gencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0й
Iencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_29_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Џ
8encoder_block_29/batch_normalization_47/FusedBatchNormV3FusedBatchNormV32encoder_block_29/max_pooling2d_29/MaxPool:output:0>encoder_block_29/batch_normalization_47/ReadVariableOp:value:0@encoder_block_29/batch_normalization_47/ReadVariableOp_1:value:0Oencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
is_training( Д
0encoder_block_30/conv2d_60/Conv2D/ReadVariableOpReadVariableOp9encoder_block_30_conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
!encoder_block_30/conv2d_60/Conv2DConv2D<encoder_block_29/batch_normalization_47/FusedBatchNormV3:y:08encoder_block_30/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides
Љ
1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_30_conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Я
"encoder_block_30/conv2d_60/BiasAddBiasAdd*encoder_block_30/conv2d_60/Conv2D:output:09encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  
encoder_block_30/conv2d_60/ReluRelu+encoder_block_30/conv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  б
)encoder_block_30/max_pooling2d_30/MaxPoolMaxPool-encoder_block_30/conv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
Г
6encoder_block_30/batch_normalization_48/ReadVariableOpReadVariableOp?encoder_block_30_batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0З
8encoder_block_30/batch_normalization_48/ReadVariableOp_1ReadVariableOpAencoder_block_30_batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0е
Gencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0й
Iencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_30_batch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Џ
8encoder_block_30/batch_normalization_48/FusedBatchNormV3FusedBatchNormV32encoder_block_30/max_pooling2d_30/MaxPool:output:0>encoder_block_30/batch_normalization_48/ReadVariableOp:value:0@encoder_block_30/batch_normalization_48/ReadVariableOp_1:value:0Oencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
is_training( Г
0encoder_block_31/conv2d_61/Conv2D/ReadVariableOpReadVariableOp9encoder_block_31_conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
!encoder_block_31/conv2d_61/Conv2DConv2D<encoder_block_30/batch_normalization_48/FusedBatchNormV3:y:08encoder_block_31/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Ј
1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_31_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
"encoder_block_31/conv2d_61/BiasAddBiasAdd*encoder_block_31/conv2d_61/Conv2D:output:09encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
encoder_block_31/conv2d_61/ReluRelu+encoder_block_31/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@а
)encoder_block_31/max_pooling2d_31/MaxPoolMaxPool-encoder_block_31/conv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides
В
6encoder_block_31/batch_normalization_49/ReadVariableOpReadVariableOp?encoder_block_31_batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8encoder_block_31/batch_normalization_49/ReadVariableOp_1ReadVariableOpAencoder_block_31_batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0д
Gencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0и
Iencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_31_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Њ
8encoder_block_31/batch_normalization_49/FusedBatchNormV3FusedBatchNormV32encoder_block_31/max_pooling2d_31/MaxPool:output:0>encoder_block_31/batch_normalization_49/ReadVariableOp:value:0@encoder_block_31/batch_normalization_49/ReadVariableOp_1:value:0Oencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( `
flatten_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
flatten_7/ReshapeReshape<encoder_block_31/batch_normalization_49/FusedBatchNormV3:y:0flatten_7/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_20/MatMulMatMulflatten_7/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџc
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype0
dense_21/MatMulMatMulflatten_7/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџc
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџi
sampling_7/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
::эЯh
sampling_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_7/strided_sliceStridedSlicesampling_7/Shape:output:0'sampling_7/strided_slice/stack:output:0)sampling_7/strided_slice/stack_1:output:0)sampling_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
sampling_7/Shape_1Shapedense_20/Relu:activations:0*
T0*
_output_shapes
::эЯj
 sampling_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sampling_7/strided_slice_1StridedSlicesampling_7/Shape_1:output:0)sampling_7/strided_slice_1/stack:output:0+sampling_7/strided_slice_1/stack_1:output:0+sampling_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
sampling_7/random_normal/shapePack!sampling_7/strided_slice:output:0#sampling_7/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_7/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_7/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Э
-sampling_7/random_normal/RandomStandardNormalRandomStandardNormal'sampling_7/random_normal/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2яж*
seedБџх)И
sampling_7/random_normal/mulMul6sampling_7/random_normal/RandomStandardNormal:output:0(sampling_7/random_normal/stddev:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
sampling_7/random_normalAddV2 sampling_7/random_normal/mul:z:0&sampling_7/random_normal/mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџU
sampling_7/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
sampling_7/mulMulsampling_7/mul/x:output:0dense_21/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ\
sampling_7/ExpExpsampling_7/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ|
sampling_7/mul_1Mulsampling_7/Exp:y:0sampling_7/random_normal:z:0*
T0*(
_output_shapes
:џџџџџџџџџ}
sampling_7/addAddV2dense_20/Relu:activations:0sampling_7/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџk
IdentityIdentitydense_20/Relu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџm

Identity_1Identitydense_21/Relu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџd

Identity_2Identitysampling_7/add:z:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOpH^encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpJ^encoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_17^encoder_block_28/batch_normalization_46/ReadVariableOp9^encoder_block_28/batch_normalization_46/ReadVariableOp_12^encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp1^encoder_block_28/conv2d_58/Conv2D/ReadVariableOpH^encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpJ^encoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_17^encoder_block_29/batch_normalization_47/ReadVariableOp9^encoder_block_29/batch_normalization_47/ReadVariableOp_12^encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp1^encoder_block_29/conv2d_59/Conv2D/ReadVariableOpH^encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpJ^encoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_17^encoder_block_30/batch_normalization_48/ReadVariableOp9^encoder_block_30/batch_normalization_48/ReadVariableOp_12^encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp1^encoder_block_30/conv2d_60/Conv2D/ReadVariableOpH^encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpJ^encoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_17^encoder_block_31/batch_normalization_49/ReadVariableOp9^encoder_block_31/batch_normalization_49/ReadVariableOp_12^encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp1^encoder_block_31/conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2
Iencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOpGencoder_block_28/batch_normalization_46/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_28/batch_normalization_46/ReadVariableOp_18encoder_block_28/batch_normalization_46/ReadVariableOp_12p
6encoder_block_28/batch_normalization_46/ReadVariableOp6encoder_block_28/batch_normalization_46/ReadVariableOp2f
1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp1encoder_block_28/conv2d_58/BiasAdd/ReadVariableOp2d
0encoder_block_28/conv2d_58/Conv2D/ReadVariableOp0encoder_block_28/conv2d_58/Conv2D/ReadVariableOp2
Iencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOpGencoder_block_29/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_29/batch_normalization_47/ReadVariableOp_18encoder_block_29/batch_normalization_47/ReadVariableOp_12p
6encoder_block_29/batch_normalization_47/ReadVariableOp6encoder_block_29/batch_normalization_47/ReadVariableOp2f
1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp1encoder_block_29/conv2d_59/BiasAdd/ReadVariableOp2d
0encoder_block_29/conv2d_59/Conv2D/ReadVariableOp0encoder_block_29/conv2d_59/Conv2D/ReadVariableOp2
Iencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOpGencoder_block_30/batch_normalization_48/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_30/batch_normalization_48/ReadVariableOp_18encoder_block_30/batch_normalization_48/ReadVariableOp_12p
6encoder_block_30/batch_normalization_48/ReadVariableOp6encoder_block_30/batch_normalization_48/ReadVariableOp2f
1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp1encoder_block_30/conv2d_60/BiasAdd/ReadVariableOp2d
0encoder_block_30/conv2d_60/Conv2D/ReadVariableOp0encoder_block_30/conv2d_60/Conv2D/ReadVariableOp2
Iencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12
Gencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOpGencoder_block_31/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_31/batch_normalization_49/ReadVariableOp_18encoder_block_31/batch_normalization_49/ReadVariableOp_12p
6encoder_block_31/batch_normalization_49/ReadVariableOp6encoder_block_31/batch_normalization_49/ReadVariableOp2f
1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp1encoder_block_31/conv2d_61/BiasAdd/ReadVariableOp2d
0encoder_block_31/conv2d_61/Conv2D/ReadVariableOp0encoder_block_31/conv2d_61/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nametensor_input
і

M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935610
input_tensorD
(conv2d_59_conv2d_readvariableop_resource:8
)conv2d_59_biasadd_readvariableop_resource:	=
.batch_normalization_47_readvariableop_resource:	?
0batch_normalization_47_readvariableop_1_resource:	N
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ6batch_normalization_47/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_47/ReadVariableOpЂ'batch_normalization_47/ReadVariableOp_1Ђ conv2d_59/BiasAdd/ReadVariableOpЂconv2d_59/Conv2D/ReadVariableOp
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_59/Conv2DConv2Dinput_tensor'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@*
paddingSAME*
strides

 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ@@m
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ@@Џ
max_pooling2d_29/MaxPoolMaxPoolconv2d_59/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ  *
ksize
*
paddingVALID*
strides

%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Щ
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_29/MaxPool:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ  :::::*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_47/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ  б
NoOpNoOp7^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ@@: : : : : : 2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ@@
&
_user_specified_nameinput_tensor
Џ7
Г
D__inference_encoder_layer_call_and_return_conditional_losses_1934613
tensor_input3
encoder_block_28_1934546:'
encoder_block_28_1934548:	'
encoder_block_28_1934550:	'
encoder_block_28_1934552:	'
encoder_block_28_1934554:	'
encoder_block_28_1934556:	4
encoder_block_29_1934559:'
encoder_block_29_1934561:	'
encoder_block_29_1934563:	'
encoder_block_29_1934565:	'
encoder_block_29_1934567:	'
encoder_block_29_1934569:	4
encoder_block_30_1934572:'
encoder_block_30_1934574:	'
encoder_block_30_1934576:	'
encoder_block_30_1934578:	'
encoder_block_30_1934580:	'
encoder_block_30_1934582:	3
encoder_block_31_1934585:@&
encoder_block_31_1934587:@&
encoder_block_31_1934589:@&
encoder_block_31_1934591:@&
encoder_block_31_1934593:@&
encoder_block_31_1934595:@$
dense_20_1934599:
 
dense_20_1934601:	$
dense_21_1934604:
 
dense_21_1934606:	
identity

identity_1

identity_2Ђ dense_20/StatefulPartitionedCallЂ dense_21/StatefulPartitionedCallЂ(encoder_block_28/StatefulPartitionedCallЂ(encoder_block_29/StatefulPartitionedCallЂ(encoder_block_30/StatefulPartitionedCallЂ(encoder_block_31/StatefulPartitionedCallЂ"sampling_7/StatefulPartitionedCall
(encoder_block_28/StatefulPartitionedCallStatefulPartitionedCalltensor_inputencoder_block_28_1934546encoder_block_28_1934548encoder_block_28_1934550encoder_block_28_1934552encoder_block_28_1934554encoder_block_28_1934556*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934259М
(encoder_block_29/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_28/StatefulPartitionedCall:output:0encoder_block_29_1934559encoder_block_29_1934561encoder_block_29_1934563encoder_block_29_1934565encoder_block_29_1934567encoder_block_29_1934569*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ  *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1934298М
(encoder_block_30/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_29/StatefulPartitionedCall:output:0encoder_block_30_1934572encoder_block_30_1934574encoder_block_30_1934576encoder_block_30_1934578encoder_block_30_1934580encoder_block_30_1934582*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934337Л
(encoder_block_31/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_30/StatefulPartitionedCall:output:0encoder_block_31_1934585encoder_block_31_1934587encoder_block_31_1934589encoder_block_31_1934591encoder_block_31_1934593encoder_block_31_1934595*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1934376ь
flatten_7/PartitionedCallPartitionedCall1encoder_block_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_flatten_7_layer_call_and_return_conditional_losses_1934164
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_20_1934599dense_20_1934601*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_1934177
 dense_21/StatefulPartitionedCallStatefulPartitionedCall"flatten_7/PartitionedCall:output:0dense_21_1934604dense_21_1934606*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_1934194Ђ
"sampling_7/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_sampling_7_layer_call_and_return_conditional_losses_1934226y
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ{

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ}

Identity_2Identity+sampling_7/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџн
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_28/StatefulPartitionedCall)^encoder_block_29/StatefulPartitionedCall)^encoder_block_30/StatefulPartitionedCall)^encoder_block_31/StatefulPartitionedCall#^sampling_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_28/StatefulPartitionedCall(encoder_block_28/StatefulPartitionedCall2T
(encoder_block_29/StatefulPartitionedCall(encoder_block_29/StatefulPartitionedCall2T
(encoder_block_30/StatefulPartitionedCall(encoder_block_30/StatefulPartitionedCall2T
(encoder_block_31/StatefulPartitionedCall(encoder_block_31/StatefulPartitionedCall2H
"sampling_7/StatefulPartitionedCall"sampling_7/StatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nametensor_input
М	

2__inference_encoder_block_30_layer_call_fn_1935627
input_tensor#
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934104x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ  : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameinput_tensor
Н	

2__inference_encoder_block_28_layer_call_fn_1935455
input_tensor"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ@@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1934024x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
Ю

S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1933967

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@А
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
 '
ё
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1934104
input_tensorD
(conv2d_60_conv2d_readvariableop_resource:8
)conv2d_60_biasadd_readvariableop_resource:	=
.batch_normalization_48_readvariableop_resource:	?
0batch_normalization_48_readvariableop_1_resource:	N
?batch_normalization_48_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource:	
identityЂ%batch_normalization_48/AssignNewValueЂ'batch_normalization_48/AssignNewValue_1Ђ6batch_normalization_48/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_48/ReadVariableOpЂ'batch_normalization_48/ReadVariableOp_1Ђ conv2d_60/BiasAdd/ReadVariableOpЂconv2d_60/Conv2D/ReadVariableOp
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Д
conv2d_60/Conv2DConv2Dinput_tensor'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  *
paddingSAME*
strides

 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ  m
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ  Џ
max_pooling2d_30/MaxPoolMaxPoolconv2d_60/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides

%batch_normalization_48/ReadVariableOpReadVariableOp.batch_normalization_48_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_48/ReadVariableOp_1ReadVariableOp0batch_normalization_48_readvariableop_1_resource*
_output_shapes	
:*
dtype0Г
6batch_normalization_48/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0З
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0з
'batch_normalization_48/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_30/MaxPool:output:0-batch_normalization_48/ReadVariableOp:value:0/batch_normalization_48/ReadVariableOp_1:value:0>batch_normalization_48/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:џџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ђ
%batch_normalization_48/AssignNewValueAssignVariableOp?batch_normalization_48_fusedbatchnormv3_readvariableop_resource4batch_normalization_48/FusedBatchNormV3:batch_mean:07^batch_normalization_48/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ќ
'batch_normalization_48/AssignNewValue_1AssignVariableOpAbatch_normalization_48_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_48/FusedBatchNormV3:batch_variance:09^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
IdentityIdentity+batch_normalization_48/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџЃ
NoOpNoOp&^batch_normalization_48/AssignNewValue(^batch_normalization_48/AssignNewValue_17^batch_normalization_48/FusedBatchNormV3/ReadVariableOp9^batch_normalization_48/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_48/ReadVariableOp(^batch_normalization_48/ReadVariableOp_1!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ  : : : : : : 2R
'batch_normalization_48/AssignNewValue_1'batch_normalization_48/AssignNewValue_12N
%batch_normalization_48/AssignNewValue%batch_normalization_48/AssignNewValue2t
8batch_normalization_48/FusedBatchNormV3/ReadVariableOp_18batch_normalization_48/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_48/FusedBatchNormV3/ReadVariableOp6batch_normalization_48/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_48/ReadVariableOp_1'batch_normalization_48/ReadVariableOp_12N
%batch_normalization_48/ReadVariableOp%batch_normalization_48/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ  
&
_user_specified_nameinput_tensor
Г

%__inference_signature_wrapper_1935046
input_1"
unknown:
	unknown_0:	
	unknown_1:	
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:	&

unknown_11:

unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	%

unknown_17:@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
 

unknown_24:	

unknown_25:
 

unknown_26:	
identity

identity_1

identity_2ЂStatefulPartitionedCallЫ
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
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__wrapped_model_1933690p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:џџџџџџџџџr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
р

M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935782
input_tensorC
(conv2d_61_conv2d_readvariableop_resource:@7
)conv2d_61_biasadd_readvariableop_resource:@<
.batch_normalization_49_readvariableop_resource:@>
0batch_normalization_49_readvariableop_1_resource:@M
?batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@
identityЂ6batch_normalization_49/FusedBatchNormV3/ReadVariableOpЂ8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1Ђ%batch_normalization_49/ReadVariableOpЂ'batch_normalization_49/ReadVariableOp_1Ђ conv2d_61/BiasAdd/ReadVariableOpЂconv2d_61/Conv2D/ReadVariableOp
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Г
conv2d_61/Conv2DConv2Dinput_tensor'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides

 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@l
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ў
max_pooling2d_31/MaxPoolMaxPoolconv2d_61/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0В
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ж
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_31/MaxPool:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ@:@:@:@:@:*
epsilon%o:*
is_training( 
IdentityIdentity+batch_normalization_49/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@б
NoOpNoOp7^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ: : : : : : 2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:џџџџџџџџџ
&
_user_specified_nameinput_tensor
Ђ	
з
8__inference_batch_normalization_48_layer_call_fn_1936045

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1933891
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
о
Ђ
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936081

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Э
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџА
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В

#__inference__traced_restore_1936440
file_prefixU
:assignvariableop_encoder_encoder_block_28_conv2d_58_kernel:I
:assignvariableop_1_encoder_encoder_block_28_conv2d_58_bias:	W
Hassignvariableop_2_encoder_encoder_block_28_batch_normalization_46_gamma:	V
Gassignvariableop_3_encoder_encoder_block_28_batch_normalization_46_beta:	]
Nassignvariableop_4_encoder_encoder_block_28_batch_normalization_46_moving_mean:	a
Rassignvariableop_5_encoder_encoder_block_28_batch_normalization_46_moving_variance:	X
<assignvariableop_6_encoder_encoder_block_29_conv2d_59_kernel:I
:assignvariableop_7_encoder_encoder_block_29_conv2d_59_bias:	W
Hassignvariableop_8_encoder_encoder_block_29_batch_normalization_47_gamma:	V
Gassignvariableop_9_encoder_encoder_block_29_batch_normalization_47_beta:	^
Oassignvariableop_10_encoder_encoder_block_29_batch_normalization_47_moving_mean:	b
Sassignvariableop_11_encoder_encoder_block_29_batch_normalization_47_moving_variance:	Y
=assignvariableop_12_encoder_encoder_block_30_conv2d_60_kernel:J
;assignvariableop_13_encoder_encoder_block_30_conv2d_60_bias:	X
Iassignvariableop_14_encoder_encoder_block_30_batch_normalization_48_gamma:	W
Hassignvariableop_15_encoder_encoder_block_30_batch_normalization_48_beta:	^
Oassignvariableop_16_encoder_encoder_block_30_batch_normalization_48_moving_mean:	b
Sassignvariableop_17_encoder_encoder_block_30_batch_normalization_48_moving_variance:	X
=assignvariableop_18_encoder_encoder_block_31_conv2d_61_kernel:@I
;assignvariableop_19_encoder_encoder_block_31_conv2d_61_bias:@W
Iassignvariableop_20_encoder_encoder_block_31_batch_normalization_49_gamma:@V
Hassignvariableop_21_encoder_encoder_block_31_batch_normalization_49_beta:@]
Oassignvariableop_22_encoder_encoder_block_31_batch_normalization_49_moving_mean:@a
Sassignvariableop_23_encoder_encoder_block_31_batch_normalization_49_moving_variance:@?
+assignvariableop_24_encoder_dense_20_kernel:
 8
)assignvariableop_25_encoder_dense_20_bias:	?
+assignvariableop_26_encoder_dense_21_kernel:
 8
)assignvariableop_27_encoder_dense_21_bias:	
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9џ	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѕ	
value	B	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B А
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOpAssignVariableOp:assignvariableop_encoder_encoder_block_28_conv2d_58_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_1AssignVariableOp:assignvariableop_1_encoder_encoder_block_28_conv2d_58_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_2AssignVariableOpHassignvariableop_2_encoder_encoder_block_28_batch_normalization_46_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_3AssignVariableOpGassignvariableop_3_encoder_encoder_block_28_batch_normalization_46_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_4AssignVariableOpNassignvariableop_4_encoder_encoder_block_28_batch_normalization_46_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:щ
AssignVariableOp_5AssignVariableOpRassignvariableop_5_encoder_encoder_block_28_batch_normalization_46_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_6AssignVariableOp<assignvariableop_6_encoder_encoder_block_29_conv2d_59_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_7AssignVariableOp:assignvariableop_7_encoder_encoder_block_29_conv2d_59_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_8AssignVariableOpHassignvariableop_8_encoder_encoder_block_29_batch_normalization_47_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_9AssignVariableOpGassignvariableop_9_encoder_encoder_block_29_batch_normalization_47_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_10AssignVariableOpOassignvariableop_10_encoder_encoder_block_29_batch_normalization_47_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_11AssignVariableOpSassignvariableop_11_encoder_encoder_block_29_batch_normalization_47_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_12AssignVariableOp=assignvariableop_12_encoder_encoder_block_30_conv2d_60_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_13AssignVariableOp;assignvariableop_13_encoder_encoder_block_30_conv2d_60_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_14AssignVariableOpIassignvariableop_14_encoder_encoder_block_30_batch_normalization_48_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:с
AssignVariableOp_15AssignVariableOpHassignvariableop_15_encoder_encoder_block_30_batch_normalization_48_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_16AssignVariableOpOassignvariableop_16_encoder_encoder_block_30_batch_normalization_48_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_17AssignVariableOpSassignvariableop_17_encoder_encoder_block_30_batch_normalization_48_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_18AssignVariableOp=assignvariableop_18_encoder_encoder_block_31_conv2d_61_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_19AssignVariableOp;assignvariableop_19_encoder_encoder_block_31_conv2d_61_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_20AssignVariableOpIassignvariableop_20_encoder_encoder_block_31_batch_normalization_49_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:с
AssignVariableOp_21AssignVariableOpHassignvariableop_21_encoder_encoder_block_31_batch_normalization_49_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_22AssignVariableOpOassignvariableop_22_encoder_encoder_block_31_batch_normalization_49_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_23AssignVariableOpSassignvariableop_23_encoder_encoder_block_31_batch_normalization_49_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_24AssignVariableOp+assignvariableop_24_encoder_dense_20_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_25AssignVariableOp)assignvariableop_25_encoder_dense_20_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_26AssignVariableOp+assignvariableop_26_encoder_dense_21_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_27AssignVariableOp)assignvariableop_27_encoder_dense_21_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 З
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: Є
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
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
	
г
8__inference_batch_normalization_49_layer_call_fn_1936104

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1933949
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ш
b
F__inference_flatten_7_layer_call_and_return_conditional_losses_1935793

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

Ц
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935919

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0л
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<Ц
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџд
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Д
serving_default 
E
input_1:
serving_default_input_1:0џџџџџџџџџ=
output_11
StatefulPartitionedCall:0џџџџџџџџџ=
output_21
StatefulPartitionedCall:1џџџџџџџџџ=
output_31
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict:Дн
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

block1

	block2


block3

block4

flattening

z_mean
z_logvar
	embedding

signatures"
_tf_keras_model
і
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21
'22
(23
)24
*25
+26
,27"
trackable_list_wrapper
Ж
0
1
2
3
4
5
6
7
8
9
10
 11
#12
$13
%14
&15
)16
*17
+18
,19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
е
2trace_0
3trace_1
4trace_2
5trace_32ъ
)__inference_encoder_layer_call_fn_1934541
)__inference_encoder_layer_call_fn_1934676
)__inference_encoder_layer_call_fn_1935111
)__inference_encoder_layer_call_fn_1935176Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 z2trace_0z3trace_1z4trace_2z5trace_3
С
6trace_0
7trace_1
8trace_2
9trace_32ж
D__inference_encoder_layer_call_and_return_conditional_losses_1934231
D__inference_encoder_layer_call_and_return_conditional_losses_1934405
D__inference_encoder_layer_call_and_return_conditional_losses_1935307
D__inference_encoder_layer_call_and_return_conditional_losses_1935438Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 z6trace_0z7trace_1z8trace_2z9trace_3
ЭBЪ
"__inference__wrapped_model_1933690input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ф
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@conv
Apooling
Bbn"
_tf_keras_layer
Ф
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jpooling
Kbn"
_tf_keras_layer
Ф
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rconv
Spooling
Tbn"
_tf_keras_layer
Ф
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[conv
\pooling
]bn"
_tf_keras_layer
Ѕ
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
Л
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
Ѕ
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
,
vserving_default"
signature_map
D:B2)encoder/encoder_block_28/conv2d_58/kernel
6:42'encoder/encoder_block_28/conv2d_58/bias
D:B25encoder/encoder_block_28/batch_normalization_46/gamma
C:A24encoder/encoder_block_28/batch_normalization_46/beta
L:J (2;encoder/encoder_block_28/batch_normalization_46/moving_mean
P:N (2?encoder/encoder_block_28/batch_normalization_46/moving_variance
E:C2)encoder/encoder_block_29/conv2d_59/kernel
6:42'encoder/encoder_block_29/conv2d_59/bias
D:B25encoder/encoder_block_29/batch_normalization_47/gamma
C:A24encoder/encoder_block_29/batch_normalization_47/beta
L:J (2;encoder/encoder_block_29/batch_normalization_47/moving_mean
P:N (2?encoder/encoder_block_29/batch_normalization_47/moving_variance
E:C2)encoder/encoder_block_30/conv2d_60/kernel
6:42'encoder/encoder_block_30/conv2d_60/bias
D:B25encoder/encoder_block_30/batch_normalization_48/gamma
C:A24encoder/encoder_block_30/batch_normalization_48/beta
L:J (2;encoder/encoder_block_30/batch_normalization_48/moving_mean
P:N (2?encoder/encoder_block_30/batch_normalization_48/moving_variance
D:B@2)encoder/encoder_block_31/conv2d_61/kernel
5:3@2'encoder/encoder_block_31/conv2d_61/bias
C:A@25encoder/encoder_block_31/batch_normalization_49/gamma
B:@@24encoder/encoder_block_31/batch_normalization_49/beta
K:I@ (2;encoder/encoder_block_31/batch_normalization_49/moving_mean
O:M@ (2?encoder/encoder_block_31/batch_normalization_49/moving_variance
+:)
 2encoder/dense_20/kernel
$:"2encoder/dense_20/bias
+:)
 2encoder/dense_21/kernel
$:"2encoder/dense_21/bias
X
0
1
2
3
!4
"5
'6
(7"
trackable_list_wrapper
X
0
	1

2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBє
)__inference_encoder_layer_call_fn_1934541input_1"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
їBє
)__inference_encoder_layer_call_fn_1934676input_1"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ќBљ
)__inference_encoder_layer_call_fn_1935111tensor_input"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ќBљ
)__inference_encoder_layer_call_fn_1935176tensor_input"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
D__inference_encoder_layer_call_and_return_conditional_losses_1934231input_1"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
D__inference_encoder_layer_call_and_return_conditional_losses_1934405input_1"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
D__inference_encoder_layer_call_and_return_conditional_losses_1935307tensor_input"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
B
D__inference_encoder_layer_call_and_return_conditional_losses_1935438tensor_input"Л
ДВА
FullArgSpec
args
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
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
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
б
|trace_0
}trace_12
2__inference_encoder_block_28_layer_call_fn_1935455
2__inference_encoder_block_28_layer_call_fn_1935472Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z|trace_0z}trace_1

~trace_0
trace_12а
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935498
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935524Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0ztrace_1
ф
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias
!_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
е
trace_0
trace_12
2__inference_encoder_block_29_layer_call_fn_1935541
2__inference_encoder_block_29_layer_call_fn_1935558Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12а
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935584
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935610Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
ф
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses

kernel
bias
!Ѓ_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ї	keras_api
Ј__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
Њ	variables
Ћtrainable_variables
Ќregularization_losses
­	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses
	Аaxis
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
В
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
е
Жtrace_0
Зtrace_12
2__inference_encoder_block_30_layer_call_fn_1935627
2__inference_encoder_block_30_layer_call_fn_1935644Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0zЗtrace_1

Иtrace_0
Йtrace_12а
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935670
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935696Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0zЙtrace_1
ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses

kernel
bias
!Р_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
	Эaxis
	gamma
 beta
!moving_mean
"moving_variance"
_tf_keras_layer
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
<
#0
$1
%2
&3"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
е
гtrace_0
дtrace_12
2__inference_encoder_block_31_layer_call_fn_1935713
2__inference_encoder_block_31_layer_call_fn_1935730Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0zдtrace_1

еtrace_0
жtrace_12а
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935756
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935782Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zеtrace_0zжtrace_1
ф
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses

#kernel
$bias
!н_jit_compiled_convolution_op"
_tf_keras_layer
Ћ
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
ф	variables
хtrainable_variables
цregularization_losses
ч	keras_api
ш__call__
+щ&call_and_return_all_conditional_losses
	ъaxis
	%gamma
&beta
'moving_mean
(moving_variance"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ч
№trace_02Ш
+__inference_flatten_7_layer_call_fn_1935787
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z№trace_0

ёtrace_02у
F__inference_flatten_7_layer_call_and_return_conditional_losses_1935793
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
ђnon_trainable_variables
ѓlayers
єmetrics
 ѕlayer_regularization_losses
іlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ц
їtrace_02Ч
*__inference_dense_20_layer_call_fn_1935802
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zїtrace_0

јtrace_02т
E__inference_dense_20_layer_call_and_return_conditional_losses_1935813
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ц
ўtrace_02Ч
*__inference_dense_21_layer_call_fn_1935822
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zўtrace_0

џtrace_02т
E__inference_dense_21_layer_call_and_return_conditional_losses_1935833
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
ш
trace_02Щ
,__inference_sampling_7_layer_call_fn_1935839
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ф
G__inference_sampling_7_layer_call_and_return_conditional_losses_1935865
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ЬBЩ
%__inference_signature_wrapper_1935046input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
љBі
2__inference_encoder_block_28_layer_call_fn_1935455input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
2__inference_encoder_block_28_layer_call_fn_1935472input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935498input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935524input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
2__inference_max_pooling2d_28_layer_call_fn_1935870
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1935875
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ч
trace_0
trace_12Ќ
8__inference_batch_normalization_46_layer_call_fn_1935888
8__inference_batch_normalization_46_layer_call_fn_1935901Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12т
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935919
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935937Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
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
љBі
2__inference_encoder_block_29_layer_call_fn_1935541input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
2__inference_encoder_block_29_layer_call_fn_1935558input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935584input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935610input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ёnon_trainable_variables
Ђlayers
Ѓmetrics
 Єlayer_regularization_losses
Ѕlayer_metrics
Є	variables
Ѕtrainable_variables
Іregularization_losses
Ј__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
ю
Іtrace_02Я
2__inference_max_pooling2d_29_layer_call_fn_1935942
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zІtrace_0

Їtrace_02ъ
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1935947
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЇtrace_0
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
Јnon_trainable_variables
Љlayers
Њmetrics
 Ћlayer_regularization_losses
Ќlayer_metrics
Њ	variables
Ћtrainable_variables
Ќregularization_losses
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
ч
­trace_0
Ўtrace_12Ќ
8__inference_batch_normalization_47_layer_call_fn_1935960
8__inference_batch_normalization_47_layer_call_fn_1935973Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z­trace_0zЎtrace_1

Џtrace_0
Аtrace_12т
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1935991
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1936009Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЏtrace_0zАtrace_1
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
љBі
2__inference_encoder_block_30_layer_call_fn_1935627input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
2__inference_encoder_block_30_layer_call_fn_1935644input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935670input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935696input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Жnon_trainable_variables
Зlayers
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
ю
Лtrace_02Я
2__inference_max_pooling2d_30_layer_call_fn_1936014
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЛtrace_0

Мtrace_02ъ
M__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_1936019
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0
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
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
ч
Тtrace_0
Уtrace_12Ќ
8__inference_batch_normalization_48_layer_call_fn_1936032
8__inference_batch_normalization_48_layer_call_fn_1936045Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0zУtrace_1

Фtrace_0
Хtrace_12т
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936063
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936081Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0zХtrace_1
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
5
[0
\1
]2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
љBі
2__inference_encoder_block_31_layer_call_fn_1935713input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
2__inference_encoder_block_31_layer_call_fn_1935730input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935756input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935782input_tensor"Џ
ЈВЄ
FullArgSpec'
args
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
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
И
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
ю
аtrace_02Я
2__inference_max_pooling2d_31_layer_call_fn_1936086
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zаtrace_0

бtrace_02ъ
M__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_1936091
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zбtrace_0
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
вnon_trainable_variables
гlayers
дmetrics
 еlayer_regularization_losses
жlayer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
ч
зtrace_0
иtrace_12Ќ
8__inference_batch_normalization_49_layer_call_fn_1936104
8__inference_batch_normalization_49_layer_call_fn_1936117Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zзtrace_0zиtrace_1

йtrace_0
кtrace_12т
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936135
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936153Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0zкtrace_1
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
еBв
+__inference_flatten_7_layer_call_fn_1935787inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
F__inference_flatten_7_layer_call_and_return_conditional_losses_1935793inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
дBб
*__inference_dense_20_layer_call_fn_1935802inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_dense_20_layer_call_and_return_conditional_losses_1935813inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
дBб
*__inference_dense_21_layer_call_fn_1935822inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_dense_21_layer_call_and_return_conditional_losses_1935833inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
тBп
,__inference_sampling_7_layer_call_fn_1935839inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
G__inference_sampling_7_layer_call_and_return_conditional_losses_1935865inputs_0inputs_1"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
2__inference_max_pooling2d_28_layer_call_fn_1935870inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1935875inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
џBќ
8__inference_batch_normalization_46_layer_call_fn_1935888inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
8__inference_batch_normalization_46_layer_call_fn_1935901inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935919inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935937inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
2__inference_max_pooling2d_29_layer_call_fn_1935942inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1935947inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
џBќ
8__inference_batch_normalization_47_layer_call_fn_1935960inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
8__inference_batch_normalization_47_layer_call_fn_1935973inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1935991inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1936009inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
2__inference_max_pooling2d_30_layer_call_fn_1936014inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
M__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_1936019inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
џBќ
8__inference_batch_normalization_48_layer_call_fn_1936032inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
8__inference_batch_normalization_48_layer_call_fn_1936045inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936063inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936081inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
2__inference_max_pooling2d_31_layer_call_fn_1936086inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
M__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_1936091inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
џBќ
8__inference_batch_normalization_49_layer_call_fn_1936104inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
8__inference_batch_normalization_49_layer_call_fn_1936117inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936135inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936153inputs"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"__inference__wrapped_model_1933690є !"#$%&'()*+,:Ђ7
0Ђ-
+(
input_1џџџџџџџџџ
Њ "Њ
/
output_1# 
output_1џџџџџџџџџ
/
output_2# 
output_2џџџџџџџџџ
/
output_3# 
output_3џџџџџџџџџћ
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935919ЃRЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ћ
S__inference_batch_normalization_46_layer_call_and_return_conditional_losses_1935937ЃRЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 е
8__inference_batch_normalization_46_layer_call_fn_1935888RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџе
8__inference_batch_normalization_46_layer_call_fn_1935901RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџћ
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1935991ЃRЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ћ
S__inference_batch_normalization_47_layer_call_and_return_conditional_losses_1936009ЃRЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 е
8__inference_batch_normalization_47_layer_call_fn_1935960RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџе
8__inference_batch_normalization_47_layer_call_fn_1935973RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџћ
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936063Ѓ !"RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ћ
S__inference_batch_normalization_48_layer_call_and_return_conditional_losses_1936081Ѓ !"RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "GЂD
=:
tensor_0,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 е
8__inference_batch_normalization_48_layer_call_fn_1936032 !"RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p

 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџе
8__inference_batch_normalization_48_layer_call_fn_1936045 !"RЂO
HЂE
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 

 
Њ "<9
unknown,џџџџџџџџџџџџџџџџџџџџџџџџџџџљ
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936135Ё%&'(QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 љ
S__inference_batch_normalization_49_layer_call_and_return_conditional_losses_1936153Ё%&'(QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ "FЂC
<9
tensor_0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 г
8__inference_batch_normalization_49_layer_call_fn_1936104%&'(QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@г
8__inference_batch_normalization_49_layer_call_fn_1936117%&'(QЂN
GЂD
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
p 

 
Њ ";8
unknown+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Ў
E__inference_dense_20_layer_call_and_return_conditional_losses_1935813e)*0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
*__inference_dense_20_layer_call_fn_1935802Z)*0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџЎ
E__inference_dense_21_layer_call_and_return_conditional_losses_1935833e+,0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 
*__inference_dense_21_layer_call_fn_1935822Z+,0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ ""
unknownџџџџџџџџџж
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935498CЂ@
9Ђ6
0-
input_tensorџџџџџџџџџ
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@@
 ж
M__inference_encoder_block_28_layer_call_and_return_conditional_losses_1935524CЂ@
9Ђ6
0-
input_tensorџџџџџџџџџ
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ@@
 Џ
2__inference_encoder_block_28_layer_call_fn_1935455yCЂ@
9Ђ6
0-
input_tensorџџџџџџџџџ
p
Њ "*'
unknownџџџџџџџџџ@@Џ
2__inference_encoder_block_28_layer_call_fn_1935472yCЂ@
9Ђ6
0-
input_tensorџџџџџџџџџ
p 
Њ "*'
unknownџџџџџџџџџ@@е
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935584BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ@@
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ  
 е
M__inference_encoder_block_29_layer_call_and_return_conditional_losses_1935610BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ@@
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ  
 Ў
2__inference_encoder_block_29_layer_call_fn_1935541xBЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ@@
p
Њ "*'
unknownџџџџџџџџџ  Ў
2__inference_encoder_block_29_layer_call_fn_1935558xBЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ@@
p 
Њ "*'
unknownџџџџџџџџџ  е
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935670 !"BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ  
p
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 е
M__inference_encoder_block_30_layer_call_and_return_conditional_losses_1935696 !"BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ  
p 
Њ "5Ђ2
+(
tensor_0џџџџџџџџџ
 Ў
2__inference_encoder_block_30_layer_call_fn_1935627x !"BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ  
p
Њ "*'
unknownџџџџџџџџџЎ
2__inference_encoder_block_30_layer_call_fn_1935644x !"BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ  
p 
Њ "*'
unknownџџџџџџџџџд
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935756#$%&'(BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 д
M__inference_encoder_block_31_layer_call_and_return_conditional_losses_1935782#$%&'(BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 ­
2__inference_encoder_block_31_layer_call_fn_1935713w#$%&'(BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ
p
Њ ")&
unknownџџџџџџџџџ@­
2__inference_encoder_block_31_layer_call_fn_1935730w#$%&'(BЂ?
8Ђ5
/,
input_tensorџџџџџџџџџ
p 
Њ ")&
unknownџџџџџџџџџ@И
D__inference_encoder_layer_call_and_return_conditional_losses_1934231я !"#$%&'()*+,JЂG
0Ђ-
+(
input_1џџџџџџџџџ
Њ

trainingp"Ђ
xu
%"

tensor_0_0џџџџџџџџџ
%"

tensor_0_1џџџџџџџџџ
%"

tensor_0_2џџџџџџџџџ
 И
D__inference_encoder_layer_call_and_return_conditional_losses_1934405я !"#$%&'()*+,JЂG
0Ђ-
+(
input_1џџџџџџџџџ
Њ

trainingp "Ђ
xu
%"

tensor_0_0џџџџџџџџџ
%"

tensor_0_1џџџџџџџџџ
%"

tensor_0_2џџџџџџџџџ
 Н
D__inference_encoder_layer_call_and_return_conditional_losses_1935307є !"#$%&'()*+,OЂL
5Ђ2
0-
tensor_inputџџџџџџџџџ
Њ

trainingp"Ђ
xu
%"

tensor_0_0џџџџџџџџџ
%"

tensor_0_1џџџџџџџџџ
%"

tensor_0_2џџџџџџџџџ
 Н
D__inference_encoder_layer_call_and_return_conditional_losses_1935438є !"#$%&'()*+,OЂL
5Ђ2
0-
tensor_inputџџџџџџџџџ
Њ

trainingp "Ђ
xu
%"

tensor_0_0џџџџџџџџџ
%"

tensor_0_1џџџџџџџџџ
%"

tensor_0_2џџџџџџџџџ
 
)__inference_encoder_layer_call_fn_1934541о !"#$%&'()*+,JЂG
0Ђ-
+(
input_1џџџџџџџџџ
Њ

trainingp"ro
# 
tensor_0џџџџџџџџџ
# 
tensor_1џџџџџџџџџ
# 
tensor_2џџџџџџџџџ
)__inference_encoder_layer_call_fn_1934676о !"#$%&'()*+,JЂG
0Ђ-
+(
input_1џџџџџџџџџ
Њ

trainingp "ro
# 
tensor_0џџџџџџџџџ
# 
tensor_1џџџџџџџџџ
# 
tensor_2џџџџџџџџџ
)__inference_encoder_layer_call_fn_1935111у !"#$%&'()*+,OЂL
5Ђ2
0-
tensor_inputџџџџџџџџџ
Њ

trainingp"ro
# 
tensor_0џџџџџџџџџ
# 
tensor_1џџџџџџџџџ
# 
tensor_2џџџџџџџџџ
)__inference_encoder_layer_call_fn_1935176у !"#$%&'()*+,OЂL
5Ђ2
0-
tensor_inputџџџџџџџџџ
Њ

trainingp "ro
# 
tensor_0џџџџџџџџџ
# 
tensor_1џџџџџџџџџ
# 
tensor_2џџџџџџџџџВ
F__inference_flatten_7_layer_call_and_return_conditional_losses_1935793h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ 
 
+__inference_flatten_7_layer_call_fn_1935787]7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ ""
unknownџџџџџџџџџ ї
M__inference_max_pooling2d_28_layer_call_and_return_conditional_losses_1935875ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_28_layer_call_fn_1935870RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_29_layer_call_and_return_conditional_losses_1935947ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_29_layer_call_fn_1935942RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_30_layer_call_and_return_conditional_losses_1936019ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_30_layer_call_fn_1936014RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџї
M__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_1936091ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 б
2__inference_max_pooling2d_31_layer_call_fn_1936086RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџй
G__inference_sampling_7_layer_call_and_return_conditional_losses_1935865\ЂY
RЂO
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
Њ "-Ђ*
# 
tensor_0џџџџџџџџџ
 Г
,__inference_sampling_7_layer_call_fn_1935839\ЂY
RЂO
MJ
# 
inputs_0џџџџџџџџџ
# 
inputs_1џџџџџџџџџ
Њ ""
unknownџџџџџџџџџЉ
%__inference_signature_wrapper_1935046џ !"#$%&'()*+,EЂB
Ђ 
;Њ8
6
input_1+(
input_1џџџџџџџџџ"Њ
/
output_1# 
output_1џџџџџџџџџ
/
output_2# 
output_2џџџџџџџџџ
/
output_3# 
output_3џџџџџџџџџ