└╝
╜М
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
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
Ы
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
resourceИ
,
Exp
x"T
y"T"
Ttype:

2
√
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
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
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
В
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
output"out_typeКэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Ж└
В
encoder/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameencoder/dense_21/bias
{
)encoder/dense_21/bias/Read/ReadVariableOpReadVariableOpencoder/dense_21/bias*
_output_shapes
:@*
dtype0
Л
encoder/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*(
shared_nameencoder/dense_21/kernel
Д
+encoder/dense_21/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_21/kernel*
_output_shapes
:	А @*
dtype0
В
encoder/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameencoder/dense_20/bias
{
)encoder/dense_20/bias/Read/ReadVariableOpReadVariableOpencoder/dense_20/bias*
_output_shapes
:@*
dtype0
Л
encoder/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А @*(
shared_nameencoder/dense_20/kernel
Д
+encoder/dense_20/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_20/kernel*
_output_shapes
:	А @*
dtype0
╓
?encoder/encoder_block_43/batch_normalization_77/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?encoder/encoder_block_43/batch_normalization_77/moving_variance
╧
Sencoder/encoder_block_43/batch_normalization_77/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_43/batch_normalization_77/moving_variance*
_output_shapes
:@*
dtype0
╬
;encoder/encoder_block_43/batch_normalization_77/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;encoder/encoder_block_43/batch_normalization_77/moving_mean
╟
Oencoder/encoder_block_43/batch_normalization_77/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_43/batch_normalization_77/moving_mean*
_output_shapes
:@*
dtype0
└
4encoder/encoder_block_43/batch_normalization_77/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64encoder/encoder_block_43/batch_normalization_77/beta
╣
Hencoder/encoder_block_43/batch_normalization_77/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_43/batch_normalization_77/beta*
_output_shapes
:@*
dtype0
┬
5encoder/encoder_block_43/batch_normalization_77/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75encoder/encoder_block_43/batch_normalization_77/gamma
╗
Iencoder/encoder_block_43/batch_normalization_77/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_43/batch_normalization_77/gamma*
_output_shapes
:@*
dtype0
ж
'encoder/encoder_block_43/conv2d_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'encoder/encoder_block_43/conv2d_89/bias
Я
;encoder/encoder_block_43/conv2d_89/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_43/conv2d_89/bias*
_output_shapes
:@*
dtype0
╖
)encoder/encoder_block_43/conv2d_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А@*:
shared_name+)encoder/encoder_block_43/conv2d_89/kernel
░
=encoder/encoder_block_43/conv2d_89/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_43/conv2d_89/kernel*'
_output_shapes
:А@*
dtype0
╫
?encoder/encoder_block_42/batch_normalization_76/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*P
shared_nameA?encoder/encoder_block_42/batch_normalization_76/moving_variance
╨
Sencoder/encoder_block_42/batch_normalization_76/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_42/batch_normalization_76/moving_variance*
_output_shapes	
:А*
dtype0
╧
;encoder/encoder_block_42/batch_normalization_76/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*L
shared_name=;encoder/encoder_block_42/batch_normalization_76/moving_mean
╚
Oencoder/encoder_block_42/batch_normalization_76/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_42/batch_normalization_76/moving_mean*
_output_shapes	
:А*
dtype0
┴
4encoder/encoder_block_42/batch_normalization_76/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*E
shared_name64encoder/encoder_block_42/batch_normalization_76/beta
║
Hencoder/encoder_block_42/batch_normalization_76/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_42/batch_normalization_76/beta*
_output_shapes	
:А*
dtype0
├
5encoder/encoder_block_42/batch_normalization_76/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*F
shared_name75encoder/encoder_block_42/batch_normalization_76/gamma
╝
Iencoder/encoder_block_42/batch_normalization_76/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_42/batch_normalization_76/gamma*
_output_shapes	
:А*
dtype0
з
'encoder/encoder_block_42/conv2d_88/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'encoder/encoder_block_42/conv2d_88/bias
а
;encoder/encoder_block_42/conv2d_88/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_42/conv2d_88/bias*
_output_shapes	
:А*
dtype0
╕
)encoder/encoder_block_42/conv2d_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*:
shared_name+)encoder/encoder_block_42/conv2d_88/kernel
▒
=encoder/encoder_block_42/conv2d_88/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_42/conv2d_88/kernel*(
_output_shapes
:АА*
dtype0
╫
?encoder/encoder_block_41/batch_normalization_75/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*P
shared_nameA?encoder/encoder_block_41/batch_normalization_75/moving_variance
╨
Sencoder/encoder_block_41/batch_normalization_75/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_41/batch_normalization_75/moving_variance*
_output_shapes	
:А*
dtype0
╧
;encoder/encoder_block_41/batch_normalization_75/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*L
shared_name=;encoder/encoder_block_41/batch_normalization_75/moving_mean
╚
Oencoder/encoder_block_41/batch_normalization_75/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_41/batch_normalization_75/moving_mean*
_output_shapes	
:А*
dtype0
┴
4encoder/encoder_block_41/batch_normalization_75/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*E
shared_name64encoder/encoder_block_41/batch_normalization_75/beta
║
Hencoder/encoder_block_41/batch_normalization_75/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_41/batch_normalization_75/beta*
_output_shapes	
:А*
dtype0
├
5encoder/encoder_block_41/batch_normalization_75/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*F
shared_name75encoder/encoder_block_41/batch_normalization_75/gamma
╝
Iencoder/encoder_block_41/batch_normalization_75/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_41/batch_normalization_75/gamma*
_output_shapes	
:А*
dtype0
з
'encoder/encoder_block_41/conv2d_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'encoder/encoder_block_41/conv2d_87/bias
а
;encoder/encoder_block_41/conv2d_87/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_41/conv2d_87/bias*
_output_shapes	
:А*
dtype0
╕
)encoder/encoder_block_41/conv2d_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*:
shared_name+)encoder/encoder_block_41/conv2d_87/kernel
▒
=encoder/encoder_block_41/conv2d_87/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_41/conv2d_87/kernel*(
_output_shapes
:АА*
dtype0
╫
?encoder/encoder_block_40/batch_normalization_74/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*P
shared_nameA?encoder/encoder_block_40/batch_normalization_74/moving_variance
╨
Sencoder/encoder_block_40/batch_normalization_74/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_40/batch_normalization_74/moving_variance*
_output_shapes	
:А*
dtype0
╧
;encoder/encoder_block_40/batch_normalization_74/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*L
shared_name=;encoder/encoder_block_40/batch_normalization_74/moving_mean
╚
Oencoder/encoder_block_40/batch_normalization_74/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_40/batch_normalization_74/moving_mean*
_output_shapes	
:А*
dtype0
┴
4encoder/encoder_block_40/batch_normalization_74/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*E
shared_name64encoder/encoder_block_40/batch_normalization_74/beta
║
Hencoder/encoder_block_40/batch_normalization_74/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_40/batch_normalization_74/beta*
_output_shapes	
:А*
dtype0
├
5encoder/encoder_block_40/batch_normalization_74/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*F
shared_name75encoder/encoder_block_40/batch_normalization_74/gamma
╝
Iencoder/encoder_block_40/batch_normalization_74/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_40/batch_normalization_74/gamma*
_output_shapes	
:А*
dtype0
з
'encoder/encoder_block_40/conv2d_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*8
shared_name)'encoder/encoder_block_40/conv2d_86/bias
а
;encoder/encoder_block_40/conv2d_86/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_40/conv2d_86/bias*
_output_shapes	
:А*
dtype0
╖
)encoder/encoder_block_40/conv2d_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*:
shared_name+)encoder/encoder_block_40/conv2d_86/kernel
░
=encoder/encoder_block_40/conv2d_86/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_40/conv2d_86/kernel*'
_output_shapes
:А*
dtype0
О
serving_default_input_1Placeholder*1
_output_shapes
:         АА*
dtype0*&
shape:         АА
Ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)encoder/encoder_block_40/conv2d_86/kernel'encoder/encoder_block_40/conv2d_86/bias5encoder/encoder_block_40/batch_normalization_74/gamma4encoder/encoder_block_40/batch_normalization_74/beta;encoder/encoder_block_40/batch_normalization_74/moving_mean?encoder/encoder_block_40/batch_normalization_74/moving_variance)encoder/encoder_block_41/conv2d_87/kernel'encoder/encoder_block_41/conv2d_87/bias5encoder/encoder_block_41/batch_normalization_75/gamma4encoder/encoder_block_41/batch_normalization_75/beta;encoder/encoder_block_41/batch_normalization_75/moving_mean?encoder/encoder_block_41/batch_normalization_75/moving_variance)encoder/encoder_block_42/conv2d_88/kernel'encoder/encoder_block_42/conv2d_88/bias5encoder/encoder_block_42/batch_normalization_76/gamma4encoder/encoder_block_42/batch_normalization_76/beta;encoder/encoder_block_42/batch_normalization_76/moving_mean?encoder/encoder_block_42/batch_normalization_76/moving_variance)encoder/encoder_block_43/conv2d_89/kernel'encoder/encoder_block_43/conv2d_89/bias5encoder/encoder_block_43/batch_normalization_77/gamma4encoder/encoder_block_43/batch_normalization_77/beta;encoder/encoder_block_43/batch_normalization_77/moving_mean?encoder/encoder_block_43/batch_normalization_77/moving_varianceencoder/dense_20/kernelencoder/dense_20/biasencoder/dense_21/kernelencoder/dense_21/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8В *.
f)R'
%__inference_signature_wrapper_2616913

NoOpNoOp
иБ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*тА
value╫АB╙А B╦А
ж
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
┌
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
Ъ
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
░
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
п
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
@conv
Apooling
Bbn*
п
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jpooling
Kbn*
п
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses
Rconv
Spooling
Tbn*
п
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[conv
\pooling
]bn*
О
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
ж
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

)kernel
*bias*
ж
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

+kernel
,bias*
О
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 

vserving_default* 
ic
VARIABLE_VALUE)encoder/encoder_block_40/conv2d_86/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'encoder/encoder_block_40/conv2d_86/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoder/encoder_block_40/batch_normalization_74/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4encoder/encoder_block_40/batch_normalization_74/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;encoder/encoder_block_40/batch_normalization_74/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?encoder/encoder_block_40/batch_normalization_74/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)encoder/encoder_block_41/conv2d_87/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'encoder/encoder_block_41/conv2d_87/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoder/encoder_block_41/batch_normalization_75/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4encoder/encoder_block_41/batch_normalization_75/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_41/batch_normalization_75/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE?encoder/encoder_block_41/batch_normalization_75/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)encoder/encoder_block_42/conv2d_88/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'encoder/encoder_block_42/conv2d_88/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5encoder/encoder_block_42/batch_normalization_76/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4encoder/encoder_block_42/batch_normalization_76/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_42/batch_normalization_76/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE?encoder/encoder_block_42/batch_normalization_76/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)encoder/encoder_block_43/conv2d_89/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'encoder/encoder_block_43/conv2d_89/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5encoder/encoder_block_43/batch_normalization_77/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4encoder/encoder_block_43/batch_normalization_77/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_43/batch_normalization_77/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE?encoder/encoder_block_43/batch_normalization_77/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
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
У
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
╧
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses

kernel
bias
!Ж_jit_compiled_convolution_op*
Ф
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses* 
▄
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
	Уaxis
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
Ш
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

Щtrace_0
Ъtrace_1* 

Ыtrace_0
Ьtrace_1* 
╧
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses

kernel
bias
!г_jit_compiled_convolution_op*
Ф
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses* 
▄
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses
	░axis
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
Ш
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

╢trace_0
╖trace_1* 

╕trace_0
╣trace_1* 
╧
║	variables
╗trainable_variables
╝regularization_losses
╜	keras_api
╛__call__
+┐&call_and_return_all_conditional_losses

kernel
bias
!└_jit_compiled_convolution_op*
Ф
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses* 
▄
╟	variables
╚trainable_variables
╔regularization_losses
╩	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses
	═axis
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
Ш
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

╙trace_0
╘trace_1* 

╒trace_0
╓trace_1* 
╧
╫	variables
╪trainable_variables
┘regularization_losses
┌	keras_api
█__call__
+▄&call_and_return_all_conditional_losses

#kernel
$bias
!▌_jit_compiled_convolution_op*
Ф
▐	variables
▀trainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses* 
▄
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
Ц
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
Ёtrace_0* 

ёtrace_0* 

)0
*1*

)0
*1*
* 
Ш
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses*

ўtrace_0* 

°trace_0* 

+0
,1*

+0
,1*
* 
Ш
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

■trace_0* 

 trace_0* 
* 
* 
* 
Ц
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 
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
Ю
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses* 

Сtrace_0* 

Тtrace_0* 
 
0
1
2
3*

0
1*
* 
Ю
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*

Шtrace_0
Щtrace_1* 

Ъtrace_0
Ыtrace_1* 
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
Ю
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses* 

жtrace_0* 

зtrace_0* 
 
0
1
2
3*

0
1*
* 
Ю
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses*

нtrace_0
оtrace_1* 

пtrace_0
░trace_1* 
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
Ю
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
║	variables
╗trainable_variables
╝regularization_losses
╛__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
┴	variables
┬trainable_variables
├regularization_losses
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses* 

╗trace_0* 

╝trace_0* 
 
0
 1
!2
"3*

0
 1*
* 
Ю
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
╟	variables
╚trainable_variables
╔regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses*

┬trace_0
├trace_1* 

─trace_0
┼trace_1* 
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
Ю
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
╫	variables
╪trainable_variables
┘regularization_losses
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
▐	variables
▀trainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses* 

╨trace_0* 

╤trace_0* 
 
%0
&1
'2
(3*

%0
&1*
* 
Ю
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses*

╫trace_0
╪trace_1* 

┘trace_0
┌trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
Щ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)encoder/encoder_block_40/conv2d_86/kernel'encoder/encoder_block_40/conv2d_86/bias5encoder/encoder_block_40/batch_normalization_74/gamma4encoder/encoder_block_40/batch_normalization_74/beta;encoder/encoder_block_40/batch_normalization_74/moving_mean?encoder/encoder_block_40/batch_normalization_74/moving_variance)encoder/encoder_block_41/conv2d_87/kernel'encoder/encoder_block_41/conv2d_87/bias5encoder/encoder_block_41/batch_normalization_75/gamma4encoder/encoder_block_41/batch_normalization_75/beta;encoder/encoder_block_41/batch_normalization_75/moving_mean?encoder/encoder_block_41/batch_normalization_75/moving_variance)encoder/encoder_block_42/conv2d_88/kernel'encoder/encoder_block_42/conv2d_88/bias5encoder/encoder_block_42/batch_normalization_76/gamma4encoder/encoder_block_42/batch_normalization_76/beta;encoder/encoder_block_42/batch_normalization_76/moving_mean?encoder/encoder_block_42/batch_normalization_76/moving_variance)encoder/encoder_block_43/conv2d_89/kernel'encoder/encoder_block_43/conv2d_89/bias5encoder/encoder_block_43/batch_normalization_77/gamma4encoder/encoder_block_43/batch_normalization_77/beta;encoder/encoder_block_43/batch_normalization_77/moving_mean?encoder/encoder_block_43/batch_normalization_77/moving_varianceencoder/dense_20/kernelencoder/dense_20/biasencoder/dense_21/kernelencoder/dense_21/biasConst*)
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
GPU2 *0J 8В *)
f$R"
 __inference__traced_save_2618213
Ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)encoder/encoder_block_40/conv2d_86/kernel'encoder/encoder_block_40/conv2d_86/bias5encoder/encoder_block_40/batch_normalization_74/gamma4encoder/encoder_block_40/batch_normalization_74/beta;encoder/encoder_block_40/batch_normalization_74/moving_mean?encoder/encoder_block_40/batch_normalization_74/moving_variance)encoder/encoder_block_41/conv2d_87/kernel'encoder/encoder_block_41/conv2d_87/bias5encoder/encoder_block_41/batch_normalization_75/gamma4encoder/encoder_block_41/batch_normalization_75/beta;encoder/encoder_block_41/batch_normalization_75/moving_mean?encoder/encoder_block_41/batch_normalization_75/moving_variance)encoder/encoder_block_42/conv2d_88/kernel'encoder/encoder_block_42/conv2d_88/bias5encoder/encoder_block_42/batch_normalization_76/gamma4encoder/encoder_block_42/batch_normalization_76/beta;encoder/encoder_block_42/batch_normalization_76/moving_mean?encoder/encoder_block_42/batch_normalization_76/moving_variance)encoder/encoder_block_43/conv2d_89/kernel'encoder/encoder_block_43/conv2d_89/bias5encoder/encoder_block_43/batch_normalization_77/gamma4encoder/encoder_block_43/batch_normalization_77/beta;encoder/encoder_block_43/batch_normalization_77/moving_mean?encoder/encoder_block_43/batch_normalization_77/moving_varianceencoder/dense_20/kernelencoder/dense_20/biasencoder/dense_21/kernelencoder/dense_21/bias*(
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
GPU2 *0J 8В *,
f'R%
#__inference__traced_restore_2618307х├
Ў
Я
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2616165
input_tensorD
(conv2d_87_conv2d_readvariableop_resource:АА8
)conv2d_87_biasadd_readvariableop_resource:	А=
.batch_normalization_75_readvariableop_resource:	А?
0batch_normalization_75_readvariableop_1_resource:	АN
?batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв6batch_normalization_75/FusedBatchNormV3/ReadVariableOpв8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_75/ReadVariableOpв'batch_normalization_75/ReadVariableOp_1в conv2d_87/BiasAdd/ReadVariableOpвconv2d_87/Conv2D/ReadVariableOpТ
conv2d_87/Conv2D/ReadVariableOpReadVariableOp(conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_87/Conv2DConv2Dinput_tensor'conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
З
 conv2d_87/BiasAdd/ReadVariableOpReadVariableOp)conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_87/BiasAddBiasAddconv2d_87/Conv2D:output:0(conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@Аm
conv2d_87/ReluReluconv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@Ап
max_pooling2d_41/MaxPoolMaxPoolconv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_75/ReadVariableOpReadVariableOp.batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_75/ReadVariableOp_1ReadVariableOp0batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╔
'batch_normalization_75/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_41/MaxPool:output:0-batch_normalization_75/ReadVariableOp:value:0/batch_normalization_75/ReadVariableOp_1:value:0>batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( Г
IdentityIdentity+batch_normalization_75/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           А╤
NoOpNoOp7^batch_normalization_75/FusedBatchNormV3/ReadVariableOp9^batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_75/ReadVariableOp(^batch_normalization_75/ReadVariableOp_1!^conv2d_87/BiasAdd/ReadVariableOp ^conv2d_87/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@А: : : : : : 2t
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_18batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_75/FusedBatchNormV3/ReadVariableOp6batch_normalization_75/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_75/ReadVariableOp_1'batch_normalization_75/ReadVariableOp_12N
%batch_normalization_75/ReadVariableOp%batch_normalization_75/ReadVariableOp2D
 conv2d_87/BiasAdd/ReadVariableOp conv2d_87/BiasAdd/ReadVariableOp2B
conv2d_87/Conv2D/ReadVariableOpconv2d_87/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@А
&
_user_specified_nameinput_tensor
а'
ё
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2615971
input_tensorD
(conv2d_88_conv2d_readvariableop_resource:АА8
)conv2d_88_biasadd_readvariableop_resource:	А=
.batch_normalization_76_readvariableop_resource:	А?
0batch_normalization_76_readvariableop_1_resource:	АN
?batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв%batch_normalization_76/AssignNewValueв'batch_normalization_76/AssignNewValue_1в6batch_normalization_76/FusedBatchNormV3/ReadVariableOpв8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_76/ReadVariableOpв'batch_normalization_76/ReadVariableOp_1в conv2d_88/BiasAdd/ReadVariableOpвconv2d_88/Conv2D/ReadVariableOpТ
conv2d_88/Conv2D/ReadVariableOpReadVariableOp(conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_88/Conv2DConv2Dinput_tensor'conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
З
 conv2d_88/BiasAdd/ReadVariableOpReadVariableOp)conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_88/BiasAddBiasAddconv2d_88/Conv2D:output:0(conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           Аm
conv2d_88/ReluReluconv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           Ап
max_pooling2d_42/MaxPoolMaxPoolconv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_76/ReadVariableOpReadVariableOp.batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_76/ReadVariableOp_1ReadVariableOp0batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╫
'batch_normalization_76/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_42/MaxPool:output:0-batch_normalization_76/ReadVariableOp:value:0/batch_normalization_76/ReadVariableOp_1:value:0>batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_76/AssignNewValueAssignVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource4batch_normalization_76/FusedBatchNormV3:batch_mean:07^batch_normalization_76/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_76/AssignNewValue_1AssignVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_76/FusedBatchNormV3:batch_variance:09^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
IdentityIdentity+batch_normalization_76/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         Аг
NoOpNoOp&^batch_normalization_76/AssignNewValue(^batch_normalization_76/AssignNewValue_17^batch_normalization_76/FusedBatchNormV3/ReadVariableOp9^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_76/ReadVariableOp(^batch_normalization_76/ReadVariableOp_1!^conv2d_88/BiasAdd/ReadVariableOp ^conv2d_88/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           А: : : : : : 2R
'batch_normalization_76/AssignNewValue_1'batch_normalization_76/AssignNewValue_12N
%batch_normalization_76/AssignNewValue%batch_normalization_76/AssignNewValue2t
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_18batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_76/FusedBatchNormV3/ReadVariableOp6batch_normalization_76/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_76/ReadVariableOp_1'batch_normalization_76/ReadVariableOp_12N
%batch_normalization_76/ReadVariableOp%batch_normalization_76/ReadVariableOp2D
 conv2d_88/BiasAdd/ReadVariableOp conv2d_88/BiasAdd/ReadVariableOp2B
conv2d_88/Conv2D/ReadVariableOpconv2d_88/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           А
&
_user_specified_nameinput_tensor
Ў
Я
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617563
input_tensorD
(conv2d_88_conv2d_readvariableop_resource:АА8
)conv2d_88_biasadd_readvariableop_resource:	А=
.batch_normalization_76_readvariableop_resource:	А?
0batch_normalization_76_readvariableop_1_resource:	АN
?batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв6batch_normalization_76/FusedBatchNormV3/ReadVariableOpв8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_76/ReadVariableOpв'batch_normalization_76/ReadVariableOp_1в conv2d_88/BiasAdd/ReadVariableOpвconv2d_88/Conv2D/ReadVariableOpТ
conv2d_88/Conv2D/ReadVariableOpReadVariableOp(conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_88/Conv2DConv2Dinput_tensor'conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
З
 conv2d_88/BiasAdd/ReadVariableOpReadVariableOp)conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_88/BiasAddBiasAddconv2d_88/Conv2D:output:0(conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           Аm
conv2d_88/ReluReluconv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           Ап
max_pooling2d_42/MaxPoolMaxPoolconv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_76/ReadVariableOpReadVariableOp.batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_76/ReadVariableOp_1ReadVariableOp0batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╔
'batch_normalization_76/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_42/MaxPool:output:0-batch_normalization_76/ReadVariableOp:value:0/batch_normalization_76/ReadVariableOp_1:value:0>batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( Г
IdentityIdentity+batch_normalization_76/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         А╤
NoOpNoOp7^batch_normalization_76/FusedBatchNormV3/ReadVariableOp9^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_76/ReadVariableOp(^batch_normalization_76/ReadVariableOp_1!^conv2d_88/BiasAdd/ReadVariableOp ^conv2d_88/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           А: : : : : : 2t
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_18batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_76/FusedBatchNormV3/ReadVariableOp6batch_normalization_76/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_76/ReadVariableOp_1'batch_normalization_76/ReadVariableOp_12N
%batch_normalization_76/ReadVariableOp%batch_normalization_76/ReadVariableOp2D
 conv2d_88/BiasAdd/ReadVariableOp conv2d_88/BiasAdd/ReadVariableOp2B
conv2d_88/Conv2D/ReadVariableOpconv2d_88/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           А
&
_user_specified_nameinput_tensor
╛	
Ш
2__inference_encoder_block_41_layer_call_fn_2617425
input_tensor#
unknown:АА
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2616165x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         @@А
&
_user_specified_nameinput_tensor
╠
Ш
*__inference_dense_20_layer_call_fn_2617669

inputs
unknown:	А @
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2616044o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
╛	
Ш
2__inference_encoder_block_42_layer_call_fn_2617511
input_tensor#
unknown:АА
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2616204x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:           А
&
_user_specified_nameinput_tensor
а	
╫
8__inference_batch_normalization_75_layer_call_fn_2617827

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2615664К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╜	
Ч
2__inference_encoder_block_40_layer_call_fn_2617322
input_tensor"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2615891x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         @@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         АА: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         АА
&
_user_specified_nameinput_tensor
┴
N
2__inference_max_pooling2d_43_layer_call_fn_2617953

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
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_2615791Г
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
Ш
╞
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617930

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
б7
л
D__inference_encoder_layer_call_and_return_conditional_losses_2616272
input_13
encoder_block_40_2616127:А'
encoder_block_40_2616129:	А'
encoder_block_40_2616131:	А'
encoder_block_40_2616133:	А'
encoder_block_40_2616135:	А'
encoder_block_40_2616137:	А4
encoder_block_41_2616166:АА'
encoder_block_41_2616168:	А'
encoder_block_41_2616170:	А'
encoder_block_41_2616172:	А'
encoder_block_41_2616174:	А'
encoder_block_41_2616176:	А4
encoder_block_42_2616205:АА'
encoder_block_42_2616207:	А'
encoder_block_42_2616209:	А'
encoder_block_42_2616211:	А'
encoder_block_42_2616213:	А'
encoder_block_42_2616215:	А3
encoder_block_43_2616244:А@&
encoder_block_43_2616246:@&
encoder_block_43_2616248:@&
encoder_block_43_2616250:@&
encoder_block_43_2616252:@&
encoder_block_43_2616254:@#
dense_20_2616258:	А @
dense_20_2616260:@#
dense_21_2616263:	А @
dense_21_2616265:@
identity

identity_1

identity_2Ив dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв(encoder_block_40/StatefulPartitionedCallв(encoder_block_41/StatefulPartitionedCallв(encoder_block_42/StatefulPartitionedCallв(encoder_block_43/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallТ
(encoder_block_40/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_block_40_2616127encoder_block_40_2616129encoder_block_40_2616131encoder_block_40_2616133encoder_block_40_2616135encoder_block_40_2616137*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2616126╝
(encoder_block_41/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_40/StatefulPartitionedCall:output:0encoder_block_41_2616166encoder_block_41_2616168encoder_block_41_2616170encoder_block_41_2616172encoder_block_41_2616174encoder_block_41_2616176*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2616165╝
(encoder_block_42/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_41/StatefulPartitionedCall:output:0encoder_block_42_2616205encoder_block_42_2616207encoder_block_42_2616209encoder_block_42_2616211encoder_block_42_2616213encoder_block_42_2616215*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2616204╗
(encoder_block_43/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_42/StatefulPartitionedCall:output:0encoder_block_43_2616244encoder_block_43_2616246encoder_block_43_2616248encoder_block_43_2616250encoder_block_43_2616252encoder_block_43_2616254*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616243ю
flatten_10/PartitionedCallPartitionedCall1encoder_block_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_2616031Х
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_2616258dense_20_2616260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2616044Х
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_21_2616263dense_21_2616265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2616061г
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sampling_10_layer_call_and_return_conditional_losses_2616093x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @z

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @}

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @▐
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_40/StatefulPartitionedCall)^encoder_block_41/StatefulPartitionedCall)^encoder_block_42/StatefulPartitionedCall)^encoder_block_43/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_40/StatefulPartitionedCall(encoder_block_40/StatefulPartitionedCall2T
(encoder_block_41/StatefulPartitionedCall(encoder_block_41/StatefulPartitionedCall2T
(encoder_block_42/StatefulPartitionedCall(encoder_block_42/StatefulPartitionedCall2T
(encoder_block_43/StatefulPartitionedCall(encoder_block_43/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall:Z V
1
_output_shapes
:         АА
!
_user_specified_name	input_1
╬
Ю
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2615834

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
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
в	
╫
8__inference_batch_normalization_76_layer_call_fn_2617912

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2615758К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
№
Ю
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2616126
input_tensorC
(conv2d_86_conv2d_readvariableop_resource:А8
)conv2d_86_biasadd_readvariableop_resource:	А=
.batch_normalization_74_readvariableop_resource:	А?
0batch_normalization_74_readvariableop_1_resource:	АN
?batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв6batch_normalization_74/FusedBatchNormV3/ReadVariableOpв8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_74/ReadVariableOpв'batch_normalization_74/ReadVariableOp_1в conv2d_86/BiasAdd/ReadVariableOpвconv2d_86/Conv2D/ReadVariableOpС
conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0╢
conv2d_86/Conv2DConv2Dinput_tensor'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
З
 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ю
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААo
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         АААп
max_pooling2d_40/MaxPoolMaxPoolconv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_74/ReadVariableOpReadVariableOp.batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_74/ReadVariableOp_1ReadVariableOp0batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╔
'batch_normalization_74/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_40/MaxPool:output:0-batch_normalization_74/ReadVariableOp:value:0/batch_normalization_74/ReadVariableOp_1:value:0>batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
is_training( Г
IdentityIdentity+batch_normalization_74/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@А╤
NoOpNoOp7^batch_normalization_74/FusedBatchNormV3/ReadVariableOp9^batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_74/ReadVariableOp(^batch_normalization_74/ReadVariableOp_1!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         АА: : : : : : 2t
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_18batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_74/FusedBatchNormV3/ReadVariableOp6batch_normalization_74/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_74/ReadVariableOp_1'batch_normalization_74/ReadVariableOp_12N
%batch_normalization_74/ReadVariableOp%batch_normalization_74/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         АА
&
_user_specified_nameinput_tensor
╢	
Т
2__inference_encoder_block_43_layer_call_fn_2617597
input_tensor"
unknown:А@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616243w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         А
&
_user_specified_nameinput_tensor
№
Ю
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617391
input_tensorC
(conv2d_86_conv2d_readvariableop_resource:А8
)conv2d_86_biasadd_readvariableop_resource:	А=
.batch_normalization_74_readvariableop_resource:	А?
0batch_normalization_74_readvariableop_1_resource:	АN
?batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв6batch_normalization_74/FusedBatchNormV3/ReadVariableOpв8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_74/ReadVariableOpв'batch_normalization_74/ReadVariableOp_1в conv2d_86/BiasAdd/ReadVariableOpвconv2d_86/Conv2D/ReadVariableOpС
conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0╢
conv2d_86/Conv2DConv2Dinput_tensor'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
З
 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ю
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААo
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         АААп
max_pooling2d_40/MaxPoolMaxPoolconv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_74/ReadVariableOpReadVariableOp.batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_74/ReadVariableOp_1ReadVariableOp0batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╔
'batch_normalization_74/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_40/MaxPool:output:0-batch_normalization_74/ReadVariableOp:value:0/batch_normalization_74/ReadVariableOp_1:value:0>batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
is_training( Г
IdentityIdentity+batch_normalization_74/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@А╤
NoOpNoOp7^batch_normalization_74/FusedBatchNormV3/ReadVariableOp9^batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_74/ReadVariableOp(^batch_normalization_74/ReadVariableOp_1!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         АА: : : : : : 2t
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_18batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_74/FusedBatchNormV3/ReadVariableOp6batch_normalization_74/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_74/ReadVariableOp_1'batch_normalization_74/ReadVariableOp_12N
%batch_normalization_74/ReadVariableOp%batch_normalization_74/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         АА
&
_user_specified_nameinput_tensor
ж'
Ё
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2615891
input_tensorC
(conv2d_86_conv2d_readvariableop_resource:А8
)conv2d_86_biasadd_readvariableop_resource:	А=
.batch_normalization_74_readvariableop_resource:	А?
0batch_normalization_74_readvariableop_1_resource:	АN
?batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв%batch_normalization_74/AssignNewValueв'batch_normalization_74/AssignNewValue_1в6batch_normalization_74/FusedBatchNormV3/ReadVariableOpв8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_74/ReadVariableOpв'batch_normalization_74/ReadVariableOp_1в conv2d_86/BiasAdd/ReadVariableOpвconv2d_86/Conv2D/ReadVariableOpС
conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0╢
conv2d_86/Conv2DConv2Dinput_tensor'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
З
 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ю
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААo
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         АААп
max_pooling2d_40/MaxPoolMaxPoolconv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_74/ReadVariableOpReadVariableOp.batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_74/ReadVariableOp_1ReadVariableOp0batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╫
'batch_normalization_74/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_40/MaxPool:output:0-batch_normalization_74/ReadVariableOp:value:0/batch_normalization_74/ReadVariableOp_1:value:0>batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_74/AssignNewValueAssignVariableOp?batch_normalization_74_fusedbatchnormv3_readvariableop_resource4batch_normalization_74/FusedBatchNormV3:batch_mean:07^batch_normalization_74/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_74/AssignNewValue_1AssignVariableOpAbatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_74/FusedBatchNormV3:batch_variance:09^batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
IdentityIdentity+batch_normalization_74/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@Аг
NoOpNoOp&^batch_normalization_74/AssignNewValue(^batch_normalization_74/AssignNewValue_17^batch_normalization_74/FusedBatchNormV3/ReadVariableOp9^batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_74/ReadVariableOp(^batch_normalization_74/ReadVariableOp_1!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         АА: : : : : : 2R
'batch_normalization_74/AssignNewValue_1'batch_normalization_74/AssignNewValue_12N
%batch_normalization_74/AssignNewValue%batch_normalization_74/AssignNewValue2t
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_18batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_74/FusedBatchNormV3/ReadVariableOp6batch_normalization_74/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_74/ReadVariableOp_1'batch_normalization_74/ReadVariableOp_12N
%batch_normalization_74/ReadVariableOp%batch_normalization_74/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         АА
&
_user_specified_nameinput_tensor
ж'
Ё
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617365
input_tensorC
(conv2d_86_conv2d_readvariableop_resource:А8
)conv2d_86_biasadd_readvariableop_resource:	А=
.batch_normalization_74_readvariableop_resource:	А?
0batch_normalization_74_readvariableop_1_resource:	АN
?batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв%batch_normalization_74/AssignNewValueв'batch_normalization_74/AssignNewValue_1в6batch_normalization_74/FusedBatchNormV3/ReadVariableOpв8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_74/ReadVariableOpв'batch_normalization_74/ReadVariableOp_1в conv2d_86/BiasAdd/ReadVariableOpвconv2d_86/Conv2D/ReadVariableOpС
conv2d_86/Conv2D/ReadVariableOpReadVariableOp(conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0╢
conv2d_86/Conv2DConv2Dinput_tensor'conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
З
 conv2d_86/BiasAdd/ReadVariableOpReadVariableOp)conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ю
conv2d_86/BiasAddBiasAddconv2d_86/Conv2D:output:0(conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААo
conv2d_86/ReluReluconv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         АААп
max_pooling2d_40/MaxPoolMaxPoolconv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_74/ReadVariableOpReadVariableOp.batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_74/ReadVariableOp_1ReadVariableOp0batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╫
'batch_normalization_74/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_40/MaxPool:output:0-batch_normalization_74/ReadVariableOp:value:0/batch_normalization_74/ReadVariableOp_1:value:0>batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_74/AssignNewValueAssignVariableOp?batch_normalization_74_fusedbatchnormv3_readvariableop_resource4batch_normalization_74/FusedBatchNormV3:batch_mean:07^batch_normalization_74/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_74/AssignNewValue_1AssignVariableOpAbatch_normalization_74_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_74/FusedBatchNormV3:batch_variance:09^batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
IdentityIdentity+batch_normalization_74/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@Аг
NoOpNoOp&^batch_normalization_74/AssignNewValue(^batch_normalization_74/AssignNewValue_17^batch_normalization_74/FusedBatchNormV3/ReadVariableOp9^batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_74/ReadVariableOp(^batch_normalization_74/ReadVariableOp_1!^conv2d_86/BiasAdd/ReadVariableOp ^conv2d_86/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         АА: : : : : : 2R
'batch_normalization_74/AssignNewValue_1'batch_normalization_74/AssignNewValue_12N
%batch_normalization_74/AssignNewValue%batch_normalization_74/AssignNewValue2t
8batch_normalization_74/FusedBatchNormV3/ReadVariableOp_18batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_74/FusedBatchNormV3/ReadVariableOp6batch_normalization_74/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_74/ReadVariableOp_1'batch_normalization_74/ReadVariableOp_12N
%batch_normalization_74/ReadVariableOp%batch_normalization_74/ReadVariableOp2D
 conv2d_86/BiasAdd/ReadVariableOp conv2d_86/BiasAdd/ReadVariableOp2B
conv2d_86/Conv2D/ReadVariableOpconv2d_86/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         АА
&
_user_specified_nameinput_tensor
Ў
Я
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2616204
input_tensorD
(conv2d_88_conv2d_readvariableop_resource:АА8
)conv2d_88_biasadd_readvariableop_resource:	А=
.batch_normalization_76_readvariableop_resource:	А?
0batch_normalization_76_readvariableop_1_resource:	АN
?batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв6batch_normalization_76/FusedBatchNormV3/ReadVariableOpв8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_76/ReadVariableOpв'batch_normalization_76/ReadVariableOp_1в conv2d_88/BiasAdd/ReadVariableOpвconv2d_88/Conv2D/ReadVariableOpТ
conv2d_88/Conv2D/ReadVariableOpReadVariableOp(conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_88/Conv2DConv2Dinput_tensor'conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
З
 conv2d_88/BiasAdd/ReadVariableOpReadVariableOp)conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_88/BiasAddBiasAddconv2d_88/Conv2D:output:0(conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           Аm
conv2d_88/ReluReluconv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           Ап
max_pooling2d_42/MaxPoolMaxPoolconv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_76/ReadVariableOpReadVariableOp.batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_76/ReadVariableOp_1ReadVariableOp0batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╔
'batch_normalization_76/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_42/MaxPool:output:0-batch_normalization_76/ReadVariableOp:value:0/batch_normalization_76/ReadVariableOp_1:value:0>batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( Г
IdentityIdentity+batch_normalization_76/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         А╤
NoOpNoOp7^batch_normalization_76/FusedBatchNormV3/ReadVariableOp9^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_76/ReadVariableOp(^batch_normalization_76/ReadVariableOp_1!^conv2d_88/BiasAdd/ReadVariableOp ^conv2d_88/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           А: : : : : : 2t
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_18batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_76/FusedBatchNormV3/ReadVariableOp6batch_normalization_76/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_76/ReadVariableOp_1'batch_normalization_76/ReadVariableOp_12N
%batch_normalization_76/ReadVariableOp%batch_normalization_76/ReadVariableOp2D
 conv2d_88/BiasAdd/ReadVariableOp conv2d_88/BiasAdd/ReadVariableOp2B
conv2d_88/Conv2D/ReadVariableOpconv2d_88/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           А
&
_user_specified_nameinput_tensor
Х
i
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2617886

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
Х
i
M__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_2617958

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
Ш	
╙
8__inference_batch_normalization_77_layer_call_fn_2617971

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallЯ
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2615816Й
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
╝	
Ш
2__inference_encoder_block_41_layer_call_fn_2617408
input_tensor#
unknown:АА
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2615931x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         @@А
&
_user_specified_nameinput_tensor
а

ў
E__inference_dense_20_layer_call_and_return_conditional_losses_2617680

inputs1
matmul_readvariableop_resource:	А @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
а

ў
E__inference_dense_21_layer_call_and_return_conditional_losses_2616061

inputs1
matmul_readvariableop_resource:	А @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
║
H
,__inference_flatten_10_layer_call_fn_2617654

inputs
identity╕
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_2616031a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
┴
N
2__inference_max_pooling2d_41_layer_call_fn_2617809

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
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2615639Г
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
ъу
д#
D__inference_encoder_layer_call_and_return_conditional_losses_2617174
tensor_inputT
9encoder_block_40_conv2d_86_conv2d_readvariableop_resource:АI
:encoder_block_40_conv2d_86_biasadd_readvariableop_resource:	АN
?encoder_block_40_batch_normalization_74_readvariableop_resource:	АP
Aencoder_block_40_batch_normalization_74_readvariableop_1_resource:	А_
Pencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	Аa
Rencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	АU
9encoder_block_41_conv2d_87_conv2d_readvariableop_resource:ААI
:encoder_block_41_conv2d_87_biasadd_readvariableop_resource:	АN
?encoder_block_41_batch_normalization_75_readvariableop_resource:	АP
Aencoder_block_41_batch_normalization_75_readvariableop_1_resource:	А_
Pencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	Аa
Rencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	АU
9encoder_block_42_conv2d_88_conv2d_readvariableop_resource:ААI
:encoder_block_42_conv2d_88_biasadd_readvariableop_resource:	АN
?encoder_block_42_batch_normalization_76_readvariableop_resource:	АP
Aencoder_block_42_batch_normalization_76_readvariableop_1_resource:	А_
Pencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	Аa
Rencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	АT
9encoder_block_43_conv2d_89_conv2d_readvariableop_resource:А@H
:encoder_block_43_conv2d_89_biasadd_readvariableop_resource:@M
?encoder_block_43_batch_normalization_77_readvariableop_resource:@O
Aencoder_block_43_batch_normalization_77_readvariableop_1_resource:@^
Pencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@`
Rencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@:
'dense_20_matmul_readvariableop_resource:	А @6
(dense_20_biasadd_readvariableop_resource:@:
'dense_21_matmul_readvariableop_resource:	А @6
(dense_21_biasadd_readvariableop_resource:@
identity

identity_1

identity_2Ивdense_20/BiasAdd/ReadVariableOpвdense_20/MatMul/ReadVariableOpвdense_21/BiasAdd/ReadVariableOpвdense_21/MatMul/ReadVariableOpв6encoder_block_40/batch_normalization_74/AssignNewValueв8encoder_block_40/batch_normalization_74/AssignNewValue_1вGencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpвIencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_40/batch_normalization_74/ReadVariableOpв8encoder_block_40/batch_normalization_74/ReadVariableOp_1в1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOpв0encoder_block_40/conv2d_86/Conv2D/ReadVariableOpв6encoder_block_41/batch_normalization_75/AssignNewValueв8encoder_block_41/batch_normalization_75/AssignNewValue_1вGencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpвIencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_41/batch_normalization_75/ReadVariableOpв8encoder_block_41/batch_normalization_75/ReadVariableOp_1в1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOpв0encoder_block_41/conv2d_87/Conv2D/ReadVariableOpв6encoder_block_42/batch_normalization_76/AssignNewValueв8encoder_block_42/batch_normalization_76/AssignNewValue_1вGencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpвIencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_42/batch_normalization_76/ReadVariableOpв8encoder_block_42/batch_normalization_76/ReadVariableOp_1в1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOpв0encoder_block_42/conv2d_88/Conv2D/ReadVariableOpв6encoder_block_43/batch_normalization_77/AssignNewValueв8encoder_block_43/batch_normalization_77/AssignNewValue_1вGencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpвIencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_43/batch_normalization_77/ReadVariableOpв8encoder_block_43/batch_normalization_77/ReadVariableOp_1в1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOpв0encoder_block_43/conv2d_89/Conv2D/ReadVariableOp│
0encoder_block_40/conv2d_86/Conv2D/ReadVariableOpReadVariableOp9encoder_block_40_conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0╪
!encoder_block_40/conv2d_86/Conv2DConv2Dtensor_input8encoder_block_40/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
й
1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_40_conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╤
"encoder_block_40/conv2d_86/BiasAddBiasAdd*encoder_block_40/conv2d_86/Conv2D:output:09encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААС
encoder_block_40/conv2d_86/ReluRelu+encoder_block_40/conv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         ААА╤
)encoder_block_40/max_pooling2d_40/MaxPoolMaxPool-encoder_block_40/conv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
│
6encoder_block_40/batch_normalization_74/ReadVariableOpReadVariableOp?encoder_block_40_batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8encoder_block_40/batch_normalization_74/ReadVariableOp_1ReadVariableOpAencoder_block_40_batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╒
Gencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┘
Iencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╜
8encoder_block_40/batch_normalization_74/FusedBatchNormV3FusedBatchNormV32encoder_block_40/max_pooling2d_40/MaxPool:output:0>encoder_block_40/batch_normalization_74/ReadVariableOp:value:0@encoder_block_40/batch_normalization_74/ReadVariableOp_1:value:0Oencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<ц
6encoder_block_40/batch_normalization_74/AssignNewValueAssignVariableOpPencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resourceEencoder_block_40/batch_normalization_74/FusedBatchNormV3:batch_mean:0H^encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ё
8encoder_block_40/batch_normalization_74/AssignNewValue_1AssignVariableOpRencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_40/batch_normalization_74/FusedBatchNormV3:batch_variance:0J^encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
0encoder_block_41/conv2d_87/Conv2D/ReadVariableOpReadVariableOp9encoder_block_41_conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
!encoder_block_41/conv2d_87/Conv2DConv2D<encoder_block_40/batch_normalization_74/FusedBatchNormV3:y:08encoder_block_41/conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
й
1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_41_conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╧
"encoder_block_41/conv2d_87/BiasAddBiasAdd*encoder_block_41/conv2d_87/Conv2D:output:09encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@АП
encoder_block_41/conv2d_87/ReluRelu+encoder_block_41/conv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@А╤
)encoder_block_41/max_pooling2d_41/MaxPoolMaxPool-encoder_block_41/conv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
│
6encoder_block_41/batch_normalization_75/ReadVariableOpReadVariableOp?encoder_block_41_batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8encoder_block_41/batch_normalization_75/ReadVariableOp_1ReadVariableOpAencoder_block_41_batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╒
Gencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┘
Iencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╜
8encoder_block_41/batch_normalization_75/FusedBatchNormV3FusedBatchNormV32encoder_block_41/max_pooling2d_41/MaxPool:output:0>encoder_block_41/batch_normalization_75/ReadVariableOp:value:0@encoder_block_41/batch_normalization_75/ReadVariableOp_1:value:0Oencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<ц
6encoder_block_41/batch_normalization_75/AssignNewValueAssignVariableOpPencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resourceEencoder_block_41/batch_normalization_75/FusedBatchNormV3:batch_mean:0H^encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ё
8encoder_block_41/batch_normalization_75/AssignNewValue_1AssignVariableOpRencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_41/batch_normalization_75/FusedBatchNormV3:batch_variance:0J^encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
0encoder_block_42/conv2d_88/Conv2D/ReadVariableOpReadVariableOp9encoder_block_42_conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
!encoder_block_42/conv2d_88/Conv2DConv2D<encoder_block_41/batch_normalization_75/FusedBatchNormV3:y:08encoder_block_42/conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
й
1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_42_conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╧
"encoder_block_42/conv2d_88/BiasAddBiasAdd*encoder_block_42/conv2d_88/Conv2D:output:09encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           АП
encoder_block_42/conv2d_88/ReluRelu+encoder_block_42/conv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           А╤
)encoder_block_42/max_pooling2d_42/MaxPoolMaxPool-encoder_block_42/conv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
│
6encoder_block_42/batch_normalization_76/ReadVariableOpReadVariableOp?encoder_block_42_batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8encoder_block_42/batch_normalization_76/ReadVariableOp_1ReadVariableOpAencoder_block_42_batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╒
Gencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┘
Iencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╜
8encoder_block_42/batch_normalization_76/FusedBatchNormV3FusedBatchNormV32encoder_block_42/max_pooling2d_42/MaxPool:output:0>encoder_block_42/batch_normalization_76/ReadVariableOp:value:0@encoder_block_42/batch_normalization_76/ReadVariableOp_1:value:0Oencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<ц
6encoder_block_42/batch_normalization_76/AssignNewValueAssignVariableOpPencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resourceEencoder_block_42/batch_normalization_76/FusedBatchNormV3:batch_mean:0H^encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ё
8encoder_block_42/batch_normalization_76/AssignNewValue_1AssignVariableOpRencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_42/batch_normalization_76/FusedBatchNormV3:batch_variance:0J^encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(│
0encoder_block_43/conv2d_89/Conv2D/ReadVariableOpReadVariableOp9encoder_block_43_conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Е
!encoder_block_43/conv2d_89/Conv2DConv2D<encoder_block_42/batch_normalization_76/FusedBatchNormV3:y:08encoder_block_43/conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
и
1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_43_conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╬
"encoder_block_43/conv2d_89/BiasAddBiasAdd*encoder_block_43/conv2d_89/Conv2D:output:09encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
encoder_block_43/conv2d_89/ReluRelu+encoder_block_43/conv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @╨
)encoder_block_43/max_pooling2d_43/MaxPoolMaxPool-encoder_block_43/conv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
▓
6encoder_block_43/batch_normalization_77/ReadVariableOpReadVariableOp?encoder_block_43_batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8encoder_block_43/batch_normalization_77/ReadVariableOp_1ReadVariableOpAencoder_block_43_batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0╘
Gencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╪
Iencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╕
8encoder_block_43/batch_normalization_77/FusedBatchNormV3FusedBatchNormV32encoder_block_43/max_pooling2d_43/MaxPool:output:0>encoder_block_43/batch_normalization_77/ReadVariableOp:value:0@encoder_block_43/batch_normalization_77/ReadVariableOp_1:value:0Oencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<ц
6encoder_block_43/batch_normalization_77/AssignNewValueAssignVariableOpPencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resourceEencoder_block_43/batch_normalization_77/FusedBatchNormV3:batch_mean:0H^encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ё
8encoder_block_43/batch_normalization_77/AssignNewValue_1AssignVariableOpRencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_43/batch_normalization_77/FusedBatchNormV3:batch_variance:0J^encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"       й
flatten_10/ReshapeReshape<encoder_block_43/batch_normalization_77/FusedBatchNormV3:y:0flatten_10/Const:output:0*
T0*(
_output_shapes
:         А З
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0Р
dense_20/MatMulMatMulflatten_10/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Д
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         @З
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0Р
dense_21/MatMulMatMulflatten_10/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Д
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         @j
sampling_10/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
::э╧i
sampling_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!sampling_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!sampling_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
sampling_10/strided_sliceStridedSlicesampling_10/Shape:output:0(sampling_10/strided_slice/stack:output:0*sampling_10/strided_slice/stack_1:output:0*sampling_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
sampling_10/Shape_1Shapedense_20/Relu:activations:0*
T0*
_output_shapes
::э╧k
!sampling_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
sampling_10/strided_slice_1StridedSlicesampling_10/Shape_1:output:0*sampling_10/strided_slice_1/stack:output:0,sampling_10/strided_slice_1/stack_1:output:0,sampling_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЯ
sampling_10/random_normal/shapePack"sampling_10/strided_slice:output:0$sampling_10/strided_slice_1:output:0*
N*
T0*
_output_shapes
:c
sampling_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    e
 sampling_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?═
.sampling_10/random_normal/RandomStandardNormalRandomStandardNormal(sampling_10/random_normal/shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2╠рg*
seed▒ х)║
sampling_10/random_normal/mulMul7sampling_10/random_normal/RandomStandardNormal:output:0)sampling_10/random_normal/stddev:output:0*
T0*'
_output_shapes
:         @а
sampling_10/random_normalAddV2!sampling_10/random_normal/mul:z:0'sampling_10/random_normal/mean:output:0*
T0*'
_output_shapes
:         @V
sampling_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Б
sampling_10/mulMulsampling_10/mul/x:output:0dense_21/Relu:activations:0*
T0*'
_output_shapes
:         @]
sampling_10/ExpExpsampling_10/mul:z:0*
T0*'
_output_shapes
:         @~
sampling_10/mul_1Mulsampling_10/Exp:y:0sampling_10/random_normal:z:0*
T0*'
_output_shapes
:         @~
sampling_10/addAddV2dense_20/Relu:activations:0sampling_10/mul_1:z:0*
T0*'
_output_shapes
:         @j
IdentityIdentitydense_20/Relu:activations:0^NoOp*
T0*'
_output_shapes
:         @l

Identity_1Identitydense_21/Relu:activations:0^NoOp*
T0*'
_output_shapes
:         @d

Identity_2Identitysampling_10/add:z:0^NoOp*
T0*'
_output_shapes
:         @р
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp7^encoder_block_40/batch_normalization_74/AssignNewValue9^encoder_block_40/batch_normalization_74/AssignNewValue_1H^encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpJ^encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_17^encoder_block_40/batch_normalization_74/ReadVariableOp9^encoder_block_40/batch_normalization_74/ReadVariableOp_12^encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp1^encoder_block_40/conv2d_86/Conv2D/ReadVariableOp7^encoder_block_41/batch_normalization_75/AssignNewValue9^encoder_block_41/batch_normalization_75/AssignNewValue_1H^encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpJ^encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_17^encoder_block_41/batch_normalization_75/ReadVariableOp9^encoder_block_41/batch_normalization_75/ReadVariableOp_12^encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp1^encoder_block_41/conv2d_87/Conv2D/ReadVariableOp7^encoder_block_42/batch_normalization_76/AssignNewValue9^encoder_block_42/batch_normalization_76/AssignNewValue_1H^encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpJ^encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_17^encoder_block_42/batch_normalization_76/ReadVariableOp9^encoder_block_42/batch_normalization_76/ReadVariableOp_12^encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp1^encoder_block_42/conv2d_88/Conv2D/ReadVariableOp7^encoder_block_43/batch_normalization_77/AssignNewValue9^encoder_block_43/batch_normalization_77/AssignNewValue_1H^encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpJ^encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_17^encoder_block_43/batch_normalization_77/ReadVariableOp9^encoder_block_43/batch_normalization_77/ReadVariableOp_12^encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp1^encoder_block_43/conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2t
8encoder_block_40/batch_normalization_74/AssignNewValue_18encoder_block_40/batch_normalization_74/AssignNewValue_12p
6encoder_block_40/batch_normalization_74/AssignNewValue6encoder_block_40/batch_normalization_74/AssignNewValue2Ц
Iencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpGencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_40/batch_normalization_74/ReadVariableOp_18encoder_block_40/batch_normalization_74/ReadVariableOp_12p
6encoder_block_40/batch_normalization_74/ReadVariableOp6encoder_block_40/batch_normalization_74/ReadVariableOp2f
1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp2d
0encoder_block_40/conv2d_86/Conv2D/ReadVariableOp0encoder_block_40/conv2d_86/Conv2D/ReadVariableOp2t
8encoder_block_41/batch_normalization_75/AssignNewValue_18encoder_block_41/batch_normalization_75/AssignNewValue_12p
6encoder_block_41/batch_normalization_75/AssignNewValue6encoder_block_41/batch_normalization_75/AssignNewValue2Ц
Iencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpGencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_41/batch_normalization_75/ReadVariableOp_18encoder_block_41/batch_normalization_75/ReadVariableOp_12p
6encoder_block_41/batch_normalization_75/ReadVariableOp6encoder_block_41/batch_normalization_75/ReadVariableOp2f
1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp2d
0encoder_block_41/conv2d_87/Conv2D/ReadVariableOp0encoder_block_41/conv2d_87/Conv2D/ReadVariableOp2t
8encoder_block_42/batch_normalization_76/AssignNewValue_18encoder_block_42/batch_normalization_76/AssignNewValue_12p
6encoder_block_42/batch_normalization_76/AssignNewValue6encoder_block_42/batch_normalization_76/AssignNewValue2Ц
Iencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpGencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_42/batch_normalization_76/ReadVariableOp_18encoder_block_42/batch_normalization_76/ReadVariableOp_12p
6encoder_block_42/batch_normalization_76/ReadVariableOp6encoder_block_42/batch_normalization_76/ReadVariableOp2f
1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp2d
0encoder_block_42/conv2d_88/Conv2D/ReadVariableOp0encoder_block_42/conv2d_88/Conv2D/ReadVariableOp2t
8encoder_block_43/batch_normalization_77/AssignNewValue_18encoder_block_43/batch_normalization_77/AssignNewValue_12p
6encoder_block_43/batch_normalization_77/AssignNewValue6encoder_block_43/batch_normalization_77/AssignNewValue2Ц
Iencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpGencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_43/batch_normalization_77/ReadVariableOp_18encoder_block_43/batch_normalization_77/ReadVariableOp_12p
6encoder_block_43/batch_normalization_77/ReadVariableOp6encoder_block_43/batch_normalization_77/ReadVariableOp2f
1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp2d
0encoder_block_43/conv2d_89/Conv2D/ReadVariableOp0encoder_block_43/conv2d_89/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         АА
&
_user_specified_nametensor_input
╝	
Ш
2__inference_encoder_block_42_layer_call_fn_2617494
input_tensor#
unknown:АА
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2615971x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:           А
&
_user_specified_nameinput_tensor
╬
Ю
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618020

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
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
Ш
╞
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2615740

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
а'
ё
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617537
input_tensorD
(conv2d_88_conv2d_readvariableop_resource:АА8
)conv2d_88_biasadd_readvariableop_resource:	А=
.batch_normalization_76_readvariableop_resource:	А?
0batch_normalization_76_readvariableop_1_resource:	АN
?batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв%batch_normalization_76/AssignNewValueв'batch_normalization_76/AssignNewValue_1в6batch_normalization_76/FusedBatchNormV3/ReadVariableOpв8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_76/ReadVariableOpв'batch_normalization_76/ReadVariableOp_1в conv2d_88/BiasAdd/ReadVariableOpвconv2d_88/Conv2D/ReadVariableOpТ
conv2d_88/Conv2D/ReadVariableOpReadVariableOp(conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_88/Conv2DConv2Dinput_tensor'conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
З
 conv2d_88/BiasAdd/ReadVariableOpReadVariableOp)conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_88/BiasAddBiasAddconv2d_88/Conv2D:output:0(conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           Аm
conv2d_88/ReluReluconv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           Ап
max_pooling2d_42/MaxPoolMaxPoolconv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_76/ReadVariableOpReadVariableOp.batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_76/ReadVariableOp_1ReadVariableOp0batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╫
'batch_normalization_76/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_42/MaxPool:output:0-batch_normalization_76/ReadVariableOp:value:0/batch_normalization_76/ReadVariableOp_1:value:0>batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_76/AssignNewValueAssignVariableOp?batch_normalization_76_fusedbatchnormv3_readvariableop_resource4batch_normalization_76/FusedBatchNormV3:batch_mean:07^batch_normalization_76/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_76/AssignNewValue_1AssignVariableOpAbatch_normalization_76_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_76/FusedBatchNormV3:batch_variance:09^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
IdentityIdentity+batch_normalization_76/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         Аг
NoOpNoOp&^batch_normalization_76/AssignNewValue(^batch_normalization_76/AssignNewValue_17^batch_normalization_76/FusedBatchNormV3/ReadVariableOp9^batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_76/ReadVariableOp(^batch_normalization_76/ReadVariableOp_1!^conv2d_88/BiasAdd/ReadVariableOp ^conv2d_88/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           А: : : : : : 2R
'batch_normalization_76/AssignNewValue_1'batch_normalization_76/AssignNewValue_12N
%batch_normalization_76/AssignNewValue%batch_normalization_76/AssignNewValue2t
8batch_normalization_76/FusedBatchNormV3/ReadVariableOp_18batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_76/FusedBatchNormV3/ReadVariableOp6batch_normalization_76/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_76/ReadVariableOp_1'batch_normalization_76/ReadVariableOp_12N
%batch_normalization_76/ReadVariableOp%batch_normalization_76/ReadVariableOp2D
 conv2d_88/BiasAdd/ReadVariableOp conv2d_88/BiasAdd/ReadVariableOp2B
conv2d_88/Conv2D/ReadVariableOpconv2d_88/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           А
&
_user_specified_nameinput_tensor
Х
i
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2617742

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
╔
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_2617660

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
и7
░
D__inference_encoder_layer_call_and_return_conditional_losses_2616345
tensor_input3
encoder_block_40_2616278:А'
encoder_block_40_2616280:	А'
encoder_block_40_2616282:	А'
encoder_block_40_2616284:	А'
encoder_block_40_2616286:	А'
encoder_block_40_2616288:	А4
encoder_block_41_2616291:АА'
encoder_block_41_2616293:	А'
encoder_block_41_2616295:	А'
encoder_block_41_2616297:	А'
encoder_block_41_2616299:	А'
encoder_block_41_2616301:	А4
encoder_block_42_2616304:АА'
encoder_block_42_2616306:	А'
encoder_block_42_2616308:	А'
encoder_block_42_2616310:	А'
encoder_block_42_2616312:	А'
encoder_block_42_2616314:	А3
encoder_block_43_2616317:А@&
encoder_block_43_2616319:@&
encoder_block_43_2616321:@&
encoder_block_43_2616323:@&
encoder_block_43_2616325:@&
encoder_block_43_2616327:@#
dense_20_2616331:	А @
dense_20_2616333:@#
dense_21_2616336:	А @
dense_21_2616338:@
identity

identity_1

identity_2Ив dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв(encoder_block_40/StatefulPartitionedCallв(encoder_block_41/StatefulPartitionedCallв(encoder_block_42/StatefulPartitionedCallв(encoder_block_43/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallХ
(encoder_block_40/StatefulPartitionedCallStatefulPartitionedCalltensor_inputencoder_block_40_2616278encoder_block_40_2616280encoder_block_40_2616282encoder_block_40_2616284encoder_block_40_2616286encoder_block_40_2616288*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2615891║
(encoder_block_41/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_40/StatefulPartitionedCall:output:0encoder_block_41_2616291encoder_block_41_2616293encoder_block_41_2616295encoder_block_41_2616297encoder_block_41_2616299encoder_block_41_2616301*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2615931║
(encoder_block_42/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_41/StatefulPartitionedCall:output:0encoder_block_42_2616304encoder_block_42_2616306encoder_block_42_2616308encoder_block_42_2616310encoder_block_42_2616312encoder_block_42_2616314*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2615971╣
(encoder_block_43/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_42/StatefulPartitionedCall:output:0encoder_block_43_2616317encoder_block_43_2616319encoder_block_43_2616321encoder_block_43_2616323encoder_block_43_2616325encoder_block_43_2616327*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616011ю
flatten_10/PartitionedCallPartitionedCall1encoder_block_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_2616031Х
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_2616331dense_20_2616333*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2616044Х
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_21_2616336dense_21_2616338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2616061г
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sampling_10_layer_call_and_return_conditional_losses_2616093x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @z

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @}

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @▐
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_40/StatefulPartitionedCall)^encoder_block_41/StatefulPartitionedCall)^encoder_block_42/StatefulPartitionedCall)^encoder_block_43/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_40/StatefulPartitionedCall(encoder_block_40/StatefulPartitionedCall2T
(encoder_block_41/StatefulPartitionedCall(encoder_block_41/StatefulPartitionedCall2T
(encoder_block_42/StatefulPartitionedCall(encoder_block_42/StatefulPartitionedCall2T
(encoder_block_43/StatefulPartitionedCall(encoder_block_43/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall:_ [
1
_output_shapes
:         АА
&
_user_specified_nametensor_input
И
┬
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618002

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
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
Б
v
-__inference_sampling_10_layer_call_fn_2617706
inputs_0
inputs_1
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sampling_10_layer_call_and_return_conditional_losses_2616093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         @:         @22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:         @
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs_0
в	
╫
8__inference_batch_normalization_75_layer_call_fn_2617840

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2615682К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ы
u
H__inference_sampling_10_layer_call_and_return_conditional_losses_2616093

inputs
inputs_1
identityИI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
Shape_1Shapeinputs*
T0*
_output_shapes
::э╧_
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
valueB:█
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
 *  А?╢
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2яжЦ*
seed▒ х)Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         @|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         @J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         @E
ExpExpmul:z:0*
T0*'
_output_shapes
:         @Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         @Q
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:         @O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         @:         @:OK
'
_output_shapes
:         @
 
_user_specified_nameinputs:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╓
в
)__inference_encoder_layer_call_fn_2616978
tensor_input"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А&

unknown_11:АА

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А%

unknown_17:А@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	А @

unknown_24:@

unknown_25:	А @

unknown_26:@
identity

identity_1

identity_2ИвStatefulPartitionedCallч
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
 *M
_output_shapes;
9:         @:         @:         @*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_2616345o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         АА
&
_user_specified_nametensor_input
Ш
╞
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617858

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ъ	
╙
8__inference_batch_normalization_77_layer_call_fn_2617984

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCallб
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
GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2615834Й
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
Ш
╞
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2615664

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
К'
ы
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617623
input_tensorC
(conv2d_89_conv2d_readvariableop_resource:А@7
)conv2d_89_biasadd_readvariableop_resource:@<
.batch_normalization_77_readvariableop_resource:@>
0batch_normalization_77_readvariableop_1_resource:@M
?batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@
identityИв%batch_normalization_77/AssignNewValueв'batch_normalization_77/AssignNewValue_1в6batch_normalization_77/FusedBatchNormV3/ReadVariableOpв8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_77/ReadVariableOpв'batch_normalization_77/ReadVariableOp_1в conv2d_89/BiasAdd/ReadVariableOpвconv2d_89/Conv2D/ReadVariableOpС
conv2d_89/Conv2D/ReadVariableOpReadVariableOp(conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0│
conv2d_89/Conv2DConv2Dinput_tensor'conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_89/BiasAdd/ReadVariableOpReadVariableOp)conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_89/BiasAddBiasAddconv2d_89/Conv2D:output:0(conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_89/ReluReluconv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @о
max_pooling2d_43/MaxPoolMaxPoolconv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Р
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╥
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_43/MaxPool:output:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_77/AssignNewValueAssignVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource4batch_normalization_77/FusedBatchNormV3:batch_mean:07^batch_normalization_77/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_77/AssignNewValue_1AssignVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_77/FusedBatchNormV3:batch_variance:09^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(В
IdentityIdentity+batch_normalization_77/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         @г
NoOpNoOp&^batch_normalization_77/AssignNewValue(^batch_normalization_77/AssignNewValue_17^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_1!^conv2d_89/BiasAdd/ReadVariableOp ^conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         А: : : : : : 2R
'batch_normalization_77/AssignNewValue_1'batch_normalization_77/AssignNewValue_12N
%batch_normalization_77/AssignNewValue%batch_normalization_77/AssignNewValue2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2D
 conv2d_89/BiasAdd/ReadVariableOp conv2d_89/BiasAdd/ReadVariableOp2B
conv2d_89/Conv2D/ReadVariableOpconv2d_89/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         А
&
_user_specified_nameinput_tensor
┴
N
2__inference_max_pooling2d_40_layer_call_fn_2617737

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
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2615563Г
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
И
┬
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2615816

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╓
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @╘
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
а	
╫
8__inference_batch_normalization_76_layer_call_fn_2617899

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2615740К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
а

ў
E__inference_dense_21_layer_call_and_return_conditional_losses_2617700

inputs1
matmul_readvariableop_resource:	А @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
┤	
Т
2__inference_encoder_block_43_layer_call_fn_2617580
input_tensor"
unknown:А@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616011w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         А
&
_user_specified_nameinput_tensor
▐
в
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2615606

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Ш
╞
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2615588

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
а

ў
E__inference_dense_20_layer_call_and_return_conditional_losses_2616044

inputs1
matmul_readvariableop_resource:	А @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
╠
Ш
*__inference_dense_21_layer_call_fn_2617689

inputs
unknown:	А @
	unknown_0:@
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2616061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А 
 
_user_specified_nameinputs
▐
в
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617804

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
▐
в
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2615682

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2617814

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
╔
c
G__inference_flatten_10_layer_call_and_return_conditional_losses_2616031

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
а'
ё
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2615931
input_tensorD
(conv2d_87_conv2d_readvariableop_resource:АА8
)conv2d_87_biasadd_readvariableop_resource:	А=
.batch_normalization_75_readvariableop_resource:	А?
0batch_normalization_75_readvariableop_1_resource:	АN
?batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв%batch_normalization_75/AssignNewValueв'batch_normalization_75/AssignNewValue_1в6batch_normalization_75/FusedBatchNormV3/ReadVariableOpв8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_75/ReadVariableOpв'batch_normalization_75/ReadVariableOp_1в conv2d_87/BiasAdd/ReadVariableOpвconv2d_87/Conv2D/ReadVariableOpТ
conv2d_87/Conv2D/ReadVariableOpReadVariableOp(conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_87/Conv2DConv2Dinput_tensor'conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
З
 conv2d_87/BiasAdd/ReadVariableOpReadVariableOp)conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_87/BiasAddBiasAddconv2d_87/Conv2D:output:0(conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@Аm
conv2d_87/ReluReluconv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@Ап
max_pooling2d_41/MaxPoolMaxPoolconv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_75/ReadVariableOpReadVariableOp.batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_75/ReadVariableOp_1ReadVariableOp0batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╫
'batch_normalization_75/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_41/MaxPool:output:0-batch_normalization_75/ReadVariableOp:value:0/batch_normalization_75/ReadVariableOp_1:value:0>batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_75/AssignNewValueAssignVariableOp?batch_normalization_75_fusedbatchnormv3_readvariableop_resource4batch_normalization_75/FusedBatchNormV3:batch_mean:07^batch_normalization_75/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_75/AssignNewValue_1AssignVariableOpAbatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_75/FusedBatchNormV3:batch_variance:09^batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
IdentityIdentity+batch_normalization_75/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           Аг
NoOpNoOp&^batch_normalization_75/AssignNewValue(^batch_normalization_75/AssignNewValue_17^batch_normalization_75/FusedBatchNormV3/ReadVariableOp9^batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_75/ReadVariableOp(^batch_normalization_75/ReadVariableOp_1!^conv2d_87/BiasAdd/ReadVariableOp ^conv2d_87/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@А: : : : : : 2R
'batch_normalization_75/AssignNewValue_1'batch_normalization_75/AssignNewValue_12N
%batch_normalization_75/AssignNewValue%batch_normalization_75/AssignNewValue2t
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_18batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_75/FusedBatchNormV3/ReadVariableOp6batch_normalization_75/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_75/ReadVariableOp_1'batch_normalization_75/ReadVariableOp_12N
%batch_normalization_75/ReadVariableOp%batch_normalization_75/ReadVariableOp2D
 conv2d_87/BiasAdd/ReadVariableOp conv2d_87/BiasAdd/ReadVariableOp2B
conv2d_87/Conv2D/ReadVariableOpconv2d_87/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@А
&
_user_specified_nameinput_tensor
р
Щ
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616243
input_tensorC
(conv2d_89_conv2d_readvariableop_resource:А@7
)conv2d_89_biasadd_readvariableop_resource:@<
.batch_normalization_77_readvariableop_resource:@>
0batch_normalization_77_readvariableop_1_resource:@M
?batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@
identityИв6batch_normalization_77/FusedBatchNormV3/ReadVariableOpв8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_77/ReadVariableOpв'batch_normalization_77/ReadVariableOp_1в conv2d_89/BiasAdd/ReadVariableOpвconv2d_89/Conv2D/ReadVariableOpС
conv2d_89/Conv2D/ReadVariableOpReadVariableOp(conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0│
conv2d_89/Conv2DConv2Dinput_tensor'conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_89/BiasAdd/ReadVariableOpReadVariableOp)conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_89/BiasAddBiasAddconv2d_89/Conv2D:output:0(conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_89/ReluReluconv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @о
max_pooling2d_43/MaxPoolMaxPoolconv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Р
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0─
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_43/MaxPool:output:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( В
IdentityIdentity+batch_normalization_77/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         @╤
NoOpNoOp7^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_1!^conv2d_89/BiasAdd/ReadVariableOp ^conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         А: : : : : : 2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2D
 conv2d_89/BiasAdd/ReadVariableOp conv2d_89/BiasAdd/ReadVariableOp2B
conv2d_89/Conv2D/ReadVariableOpconv2d_89/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         А
&
_user_specified_nameinput_tensor
й
Щ
%__inference_signature_wrapper_2616913
input_1"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А&

unknown_11:АА

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А%

unknown_17:А@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	А @

unknown_24:@

unknown_25:	А @

unknown_26:@
identity

identity_1

identity_2ИвStatefulPartitionedCall╚
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
 *M
_output_shapes;
9:         @:         @:         @*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8В *+
f&R$
"__inference__wrapped_model_2615557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         АА
!
_user_specified_name	input_1
┴
N
2__inference_max_pooling2d_42_layer_call_fn_2617881

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
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2615715Г
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
К'
ы
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616011
input_tensorC
(conv2d_89_conv2d_readvariableop_resource:А@7
)conv2d_89_biasadd_readvariableop_resource:@<
.batch_normalization_77_readvariableop_resource:@>
0batch_normalization_77_readvariableop_1_resource:@M
?batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@
identityИв%batch_normalization_77/AssignNewValueв'batch_normalization_77/AssignNewValue_1в6batch_normalization_77/FusedBatchNormV3/ReadVariableOpв8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_77/ReadVariableOpв'batch_normalization_77/ReadVariableOp_1в conv2d_89/BiasAdd/ReadVariableOpвconv2d_89/Conv2D/ReadVariableOpС
conv2d_89/Conv2D/ReadVariableOpReadVariableOp(conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0│
conv2d_89/Conv2DConv2Dinput_tensor'conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_89/BiasAdd/ReadVariableOpReadVariableOp)conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_89/BiasAddBiasAddconv2d_89/Conv2D:output:0(conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_89/ReluReluconv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @о
max_pooling2d_43/MaxPoolMaxPoolconv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Р
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╥
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_43/MaxPool:output:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_77/AssignNewValueAssignVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource4batch_normalization_77/FusedBatchNormV3:batch_mean:07^batch_normalization_77/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_77/AssignNewValue_1AssignVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_77/FusedBatchNormV3:batch_variance:09^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(В
IdentityIdentity+batch_normalization_77/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         @г
NoOpNoOp&^batch_normalization_77/AssignNewValue(^batch_normalization_77/AssignNewValue_17^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_1!^conv2d_89/BiasAdd/ReadVariableOp ^conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         А: : : : : : 2R
'batch_normalization_77/AssignNewValue_1'batch_normalization_77/AssignNewValue_12N
%batch_normalization_77/AssignNewValue%batch_normalization_77/AssignNewValue2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2D
 conv2d_89/BiasAdd/ReadVariableOp conv2d_89/BiasAdd/ReadVariableOp2B
conv2d_89/Conv2D/ReadVariableOpconv2d_89/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         А
&
_user_specified_nameinput_tensor
▐
в
)__inference_encoder_layer_call_fn_2617043
tensor_input"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А&

unknown_11:АА

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А%

unknown_17:А@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	А @

unknown_24:@

unknown_25:	А @

unknown_26:@
identity

identity_1

identity_2ИвStatefulPartitionedCallя
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
 *M
_output_shapes;
9:         @:         @:         @*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_2616480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         АА
&
_user_specified_nametensor_input
┐	
Ч
2__inference_encoder_block_40_layer_call_fn_2617339
input_tensor"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2616126x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         @@А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         АА: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         АА
&
_user_specified_nameinput_tensor
Щ7
л
D__inference_encoder_layer_call_and_return_conditional_losses_2616098
input_13
encoder_block_40_2615892:А'
encoder_block_40_2615894:	А'
encoder_block_40_2615896:	А'
encoder_block_40_2615898:	А'
encoder_block_40_2615900:	А'
encoder_block_40_2615902:	А4
encoder_block_41_2615932:АА'
encoder_block_41_2615934:	А'
encoder_block_41_2615936:	А'
encoder_block_41_2615938:	А'
encoder_block_41_2615940:	А'
encoder_block_41_2615942:	А4
encoder_block_42_2615972:АА'
encoder_block_42_2615974:	А'
encoder_block_42_2615976:	А'
encoder_block_42_2615978:	А'
encoder_block_42_2615980:	А'
encoder_block_42_2615982:	А3
encoder_block_43_2616012:А@&
encoder_block_43_2616014:@&
encoder_block_43_2616016:@&
encoder_block_43_2616018:@&
encoder_block_43_2616020:@&
encoder_block_43_2616022:@#
dense_20_2616045:	А @
dense_20_2616047:@#
dense_21_2616062:	А @
dense_21_2616064:@
identity

identity_1

identity_2Ив dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв(encoder_block_40/StatefulPartitionedCallв(encoder_block_41/StatefulPartitionedCallв(encoder_block_42/StatefulPartitionedCallв(encoder_block_43/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallР
(encoder_block_40/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_block_40_2615892encoder_block_40_2615894encoder_block_40_2615896encoder_block_40_2615898encoder_block_40_2615900encoder_block_40_2615902*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2615891║
(encoder_block_41/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_40/StatefulPartitionedCall:output:0encoder_block_41_2615932encoder_block_41_2615934encoder_block_41_2615936encoder_block_41_2615938encoder_block_41_2615940encoder_block_41_2615942*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2615931║
(encoder_block_42/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_41/StatefulPartitionedCall:output:0encoder_block_42_2615972encoder_block_42_2615974encoder_block_42_2615976encoder_block_42_2615978encoder_block_42_2615980encoder_block_42_2615982*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2615971╣
(encoder_block_43/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_42/StatefulPartitionedCall:output:0encoder_block_43_2616012encoder_block_43_2616014encoder_block_43_2616016encoder_block_43_2616018encoder_block_43_2616020encoder_block_43_2616022*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616011ю
flatten_10/PartitionedCallPartitionedCall1encoder_block_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_2616031Х
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_2616045dense_20_2616047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2616044Х
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_21_2616062dense_21_2616064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2616061г
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sampling_10_layer_call_and_return_conditional_losses_2616093x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @z

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @}

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @▐
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_40/StatefulPartitionedCall)^encoder_block_41/StatefulPartitionedCall)^encoder_block_42/StatefulPartitionedCall)^encoder_block_43/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_40/StatefulPartitionedCall(encoder_block_40/StatefulPartitionedCall2T
(encoder_block_41/StatefulPartitionedCall(encoder_block_41/StatefulPartitionedCall2T
(encoder_block_42/StatefulPartitionedCall(encoder_block_42/StatefulPartitionedCall2T
(encoder_block_43/StatefulPartitionedCall(encoder_block_43/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall:Z V
1
_output_shapes
:         АА
!
_user_specified_name	input_1
оВ
¤
#__inference__traced_restore_2618307
file_prefixU
:assignvariableop_encoder_encoder_block_40_conv2d_86_kernel:АI
:assignvariableop_1_encoder_encoder_block_40_conv2d_86_bias:	АW
Hassignvariableop_2_encoder_encoder_block_40_batch_normalization_74_gamma:	АV
Gassignvariableop_3_encoder_encoder_block_40_batch_normalization_74_beta:	А]
Nassignvariableop_4_encoder_encoder_block_40_batch_normalization_74_moving_mean:	Аa
Rassignvariableop_5_encoder_encoder_block_40_batch_normalization_74_moving_variance:	АX
<assignvariableop_6_encoder_encoder_block_41_conv2d_87_kernel:ААI
:assignvariableop_7_encoder_encoder_block_41_conv2d_87_bias:	АW
Hassignvariableop_8_encoder_encoder_block_41_batch_normalization_75_gamma:	АV
Gassignvariableop_9_encoder_encoder_block_41_batch_normalization_75_beta:	А^
Oassignvariableop_10_encoder_encoder_block_41_batch_normalization_75_moving_mean:	Аb
Sassignvariableop_11_encoder_encoder_block_41_batch_normalization_75_moving_variance:	АY
=assignvariableop_12_encoder_encoder_block_42_conv2d_88_kernel:ААJ
;assignvariableop_13_encoder_encoder_block_42_conv2d_88_bias:	АX
Iassignvariableop_14_encoder_encoder_block_42_batch_normalization_76_gamma:	АW
Hassignvariableop_15_encoder_encoder_block_42_batch_normalization_76_beta:	А^
Oassignvariableop_16_encoder_encoder_block_42_batch_normalization_76_moving_mean:	Аb
Sassignvariableop_17_encoder_encoder_block_42_batch_normalization_76_moving_variance:	АX
=assignvariableop_18_encoder_encoder_block_43_conv2d_89_kernel:А@I
;assignvariableop_19_encoder_encoder_block_43_conv2d_89_bias:@W
Iassignvariableop_20_encoder_encoder_block_43_batch_normalization_77_gamma:@V
Hassignvariableop_21_encoder_encoder_block_43_batch_normalization_77_beta:@]
Oassignvariableop_22_encoder_encoder_block_43_batch_normalization_77_moving_mean:@a
Sassignvariableop_23_encoder_encoder_block_43_batch_normalization_77_moving_variance:@>
+assignvariableop_24_encoder_dense_20_kernel:	А @7
)assignvariableop_25_encoder_dense_20_bias:@>
+assignvariableop_26_encoder_dense_21_kernel:	А @7
)assignvariableop_27_encoder_dense_21_bias:@
identity_29ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9 	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*е	
valueЫ	BШ	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHк
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ░
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*И
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOpAssignVariableOp:assignvariableop_encoder_encoder_block_40_conv2d_86_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_1AssignVariableOp:assignvariableop_1_encoder_encoder_block_40_conv2d_86_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_2AssignVariableOpHassignvariableop_2_encoder_encoder_block_40_batch_normalization_74_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:▐
AssignVariableOp_3AssignVariableOpGassignvariableop_3_encoder_encoder_block_40_batch_normalization_74_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_4AssignVariableOpNassignvariableop_4_encoder_encoder_block_40_batch_normalization_74_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:щ
AssignVariableOp_5AssignVariableOpRassignvariableop_5_encoder_encoder_block_40_batch_normalization_74_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╙
AssignVariableOp_6AssignVariableOp<assignvariableop_6_encoder_encoder_block_41_conv2d_87_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╤
AssignVariableOp_7AssignVariableOp:assignvariableop_7_encoder_encoder_block_41_conv2d_87_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_8AssignVariableOpHassignvariableop_8_encoder_encoder_block_41_batch_normalization_75_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:▐
AssignVariableOp_9AssignVariableOpGassignvariableop_9_encoder_encoder_block_41_batch_normalization_75_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_10AssignVariableOpOassignvariableop_10_encoder_encoder_block_41_batch_normalization_75_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_11AssignVariableOpSassignvariableop_11_encoder_encoder_block_41_batch_normalization_75_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╓
AssignVariableOp_12AssignVariableOp=assignvariableop_12_encoder_encoder_block_42_conv2d_88_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_13AssignVariableOp;assignvariableop_13_encoder_encoder_block_42_conv2d_88_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_14AssignVariableOpIassignvariableop_14_encoder_encoder_block_42_batch_normalization_76_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:с
AssignVariableOp_15AssignVariableOpHassignvariableop_15_encoder_encoder_block_42_batch_normalization_76_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_16AssignVariableOpOassignvariableop_16_encoder_encoder_block_42_batch_normalization_76_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_17AssignVariableOpSassignvariableop_17_encoder_encoder_block_42_batch_normalization_76_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╓
AssignVariableOp_18AssignVariableOp=assignvariableop_18_encoder_encoder_block_43_conv2d_89_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╘
AssignVariableOp_19AssignVariableOp;assignvariableop_19_encoder_encoder_block_43_conv2d_89_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_20AssignVariableOpIassignvariableop_20_encoder_encoder_block_43_batch_normalization_77_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:с
AssignVariableOp_21AssignVariableOpHassignvariableop_21_encoder_encoder_block_43_batch_normalization_77_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:ш
AssignVariableOp_22AssignVariableOpOassignvariableop_22_encoder_encoder_block_43_batch_normalization_77_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ь
AssignVariableOp_23AssignVariableOpSassignvariableop_23_encoder_encoder_block_43_batch_normalization_77_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_24AssignVariableOp+assignvariableop_24_encoder_dense_20_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_25AssignVariableOp)assignvariableop_25_encoder_dense_20_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_26AssignVariableOp+assignvariableop_26_encoder_dense_21_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_27AssignVariableOp)assignvariableop_27_encoder_dense_21_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╖
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: д
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
Ф╥
э"
"__inference__wrapped_model_2615557
input_1\
Aencoder_encoder_block_40_conv2d_86_conv2d_readvariableop_resource:АQ
Bencoder_encoder_block_40_conv2d_86_biasadd_readvariableop_resource:	АV
Gencoder_encoder_block_40_batch_normalization_74_readvariableop_resource:	АX
Iencoder_encoder_block_40_batch_normalization_74_readvariableop_1_resource:	Аg
Xencoder_encoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	Аi
Zencoder_encoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	А]
Aencoder_encoder_block_41_conv2d_87_conv2d_readvariableop_resource:ААQ
Bencoder_encoder_block_41_conv2d_87_biasadd_readvariableop_resource:	АV
Gencoder_encoder_block_41_batch_normalization_75_readvariableop_resource:	АX
Iencoder_encoder_block_41_batch_normalization_75_readvariableop_1_resource:	Аg
Xencoder_encoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	Аi
Zencoder_encoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	А]
Aencoder_encoder_block_42_conv2d_88_conv2d_readvariableop_resource:ААQ
Bencoder_encoder_block_42_conv2d_88_biasadd_readvariableop_resource:	АV
Gencoder_encoder_block_42_batch_normalization_76_readvariableop_resource:	АX
Iencoder_encoder_block_42_batch_normalization_76_readvariableop_1_resource:	Аg
Xencoder_encoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	Аi
Zencoder_encoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	А\
Aencoder_encoder_block_43_conv2d_89_conv2d_readvariableop_resource:А@P
Bencoder_encoder_block_43_conv2d_89_biasadd_readvariableop_resource:@U
Gencoder_encoder_block_43_batch_normalization_77_readvariableop_resource:@W
Iencoder_encoder_block_43_batch_normalization_77_readvariableop_1_resource:@f
Xencoder_encoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@h
Zencoder_encoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@B
/encoder_dense_20_matmul_readvariableop_resource:	А @>
0encoder_dense_20_biasadd_readvariableop_resource:@B
/encoder_dense_21_matmul_readvariableop_resource:	А @>
0encoder_dense_21_biasadd_readvariableop_resource:@
identity

identity_1

identity_2Ив'encoder/dense_20/BiasAdd/ReadVariableOpв&encoder/dense_20/MatMul/ReadVariableOpв'encoder/dense_21/BiasAdd/ReadVariableOpв&encoder/dense_21/MatMul/ReadVariableOpвOencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpвQencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в>encoder/encoder_block_40/batch_normalization_74/ReadVariableOpв@encoder/encoder_block_40/batch_normalization_74/ReadVariableOp_1в9encoder/encoder_block_40/conv2d_86/BiasAdd/ReadVariableOpв8encoder/encoder_block_40/conv2d_86/Conv2D/ReadVariableOpвOencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpвQencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в>encoder/encoder_block_41/batch_normalization_75/ReadVariableOpв@encoder/encoder_block_41/batch_normalization_75/ReadVariableOp_1в9encoder/encoder_block_41/conv2d_87/BiasAdd/ReadVariableOpв8encoder/encoder_block_41/conv2d_87/Conv2D/ReadVariableOpвOencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpвQencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в>encoder/encoder_block_42/batch_normalization_76/ReadVariableOpв@encoder/encoder_block_42/batch_normalization_76/ReadVariableOp_1в9encoder/encoder_block_42/conv2d_88/BiasAdd/ReadVariableOpв8encoder/encoder_block_42/conv2d_88/Conv2D/ReadVariableOpвOencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpвQencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в>encoder/encoder_block_43/batch_normalization_77/ReadVariableOpв@encoder/encoder_block_43/batch_normalization_77/ReadVariableOp_1в9encoder/encoder_block_43/conv2d_89/BiasAdd/ReadVariableOpв8encoder/encoder_block_43/conv2d_89/Conv2D/ReadVariableOp├
8encoder/encoder_block_40/conv2d_86/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_40_conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0у
)encoder/encoder_block_40/conv2d_86/Conv2DConv2Dinput_1@encoder/encoder_block_40/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
╣
9encoder/encoder_block_40/conv2d_86/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_40_conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0щ
*encoder/encoder_block_40/conv2d_86/BiasAddBiasAdd2encoder/encoder_block_40/conv2d_86/Conv2D:output:0Aencoder/encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААб
'encoder/encoder_block_40/conv2d_86/ReluRelu3encoder/encoder_block_40/conv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         АААс
1encoder/encoder_block_40/max_pooling2d_40/MaxPoolMaxPool5encoder/encoder_block_40/conv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_40/batch_normalization_74/ReadVariableOpReadVariableOpGencoder_encoder_block_40_batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0╟
@encoder/encoder_block_40/batch_normalization_74/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_40_batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0х
Oencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0щ
Qencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▀
@encoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_40/max_pooling2d_40/MaxPool:output:0Fencoder/encoder_block_40/batch_normalization_74/ReadVariableOp:value:0Hencoder/encoder_block_40/batch_normalization_74/ReadVariableOp_1:value:0Wencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
is_training( ─
8encoder/encoder_block_41/conv2d_87/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_41_conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ю
)encoder/encoder_block_41/conv2d_87/Conv2DConv2DDencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3:y:0@encoder/encoder_block_41/conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
╣
9encoder/encoder_block_41/conv2d_87/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_41_conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ч
*encoder/encoder_block_41/conv2d_87/BiasAddBiasAdd2encoder/encoder_block_41/conv2d_87/Conv2D:output:0Aencoder/encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@АЯ
'encoder/encoder_block_41/conv2d_87/ReluRelu3encoder/encoder_block_41/conv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@Ас
1encoder/encoder_block_41/max_pooling2d_41/MaxPoolMaxPool5encoder/encoder_block_41/conv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_41/batch_normalization_75/ReadVariableOpReadVariableOpGencoder_encoder_block_41_batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0╟
@encoder/encoder_block_41/batch_normalization_75/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_41_batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0х
Oencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0щ
Qencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▀
@encoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_41/max_pooling2d_41/MaxPool:output:0Fencoder/encoder_block_41/batch_normalization_75/ReadVariableOp:value:0Hencoder/encoder_block_41/batch_normalization_75/ReadVariableOp_1:value:0Wencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( ─
8encoder/encoder_block_42/conv2d_88/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_42_conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ю
)encoder/encoder_block_42/conv2d_88/Conv2DConv2DDencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3:y:0@encoder/encoder_block_42/conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
╣
9encoder/encoder_block_42/conv2d_88/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_42_conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0ч
*encoder/encoder_block_42/conv2d_88/BiasAddBiasAdd2encoder/encoder_block_42/conv2d_88/Conv2D:output:0Aencoder/encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           АЯ
'encoder/encoder_block_42/conv2d_88/ReluRelu3encoder/encoder_block_42/conv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           Ас
1encoder/encoder_block_42/max_pooling2d_42/MaxPoolMaxPool5encoder/encoder_block_42/conv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_42/batch_normalization_76/ReadVariableOpReadVariableOpGencoder_encoder_block_42_batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0╟
@encoder/encoder_block_42/batch_normalization_76/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_42_batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0х
Oencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0щ
Qencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0▀
@encoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_42/max_pooling2d_42/MaxPool:output:0Fencoder/encoder_block_42/batch_normalization_76/ReadVariableOp:value:0Hencoder/encoder_block_42/batch_normalization_76/ReadVariableOp_1:value:0Wencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( ├
8encoder/encoder_block_43/conv2d_89/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_43_conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Э
)encoder/encoder_block_43/conv2d_89/Conv2DConv2DDencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3:y:0@encoder/encoder_block_43/conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
╕
9encoder/encoder_block_43/conv2d_89/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_43_conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ц
*encoder/encoder_block_43/conv2d_89/BiasAddBiasAdd2encoder/encoder_block_43/conv2d_89/Conv2D:output:0Aencoder/encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @Ю
'encoder/encoder_block_43/conv2d_89/ReluRelu3encoder/encoder_block_43/conv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @р
1encoder/encoder_block_43/max_pooling2d_43/MaxPoolMaxPool5encoder/encoder_block_43/conv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
┬
>encoder/encoder_block_43/batch_normalization_77/ReadVariableOpReadVariableOpGencoder_encoder_block_43_batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0╞
@encoder/encoder_block_43/batch_normalization_77/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_43_batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0ф
Oencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ш
Qencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┌
@encoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_43/max_pooling2d_43/MaxPool:output:0Fencoder/encoder_block_43/batch_normalization_77/ReadVariableOp:value:0Hencoder/encoder_block_43/batch_normalization_77/ReadVariableOp_1:value:0Wencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( i
encoder/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ┴
encoder/flatten_10/ReshapeReshapeDencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3:y:0!encoder/flatten_10/Const:output:0*
T0*(
_output_shapes
:         А Ч
&encoder/dense_20/MatMul/ReadVariableOpReadVariableOp/encoder_dense_20_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0и
encoder/dense_20/MatMulMatMul#encoder/flatten_10/Reshape:output:0.encoder/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ф
'encoder/dense_20/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0й
encoder/dense_20/BiasAddBiasAdd!encoder/dense_20/MatMul:product:0/encoder/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
encoder/dense_20/ReluRelu!encoder/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         @Ч
&encoder/dense_21/MatMul/ReadVariableOpReadVariableOp/encoder_dense_21_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0и
encoder/dense_21/MatMulMatMul#encoder/flatten_10/Reshape:output:0.encoder/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ф
'encoder/dense_21/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0й
encoder/dense_21/BiasAddBiasAdd!encoder/dense_21/MatMul:product:0/encoder/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
encoder/dense_21/ReluRelu!encoder/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         @z
encoder/sampling_10/ShapeShape#encoder/dense_20/Relu:activations:0*
T0*
_output_shapes
::э╧q
'encoder/sampling_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)encoder/sampling_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)encoder/sampling_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!encoder/sampling_10/strided_sliceStridedSlice"encoder/sampling_10/Shape:output:00encoder/sampling_10/strided_slice/stack:output:02encoder/sampling_10/strided_slice/stack_1:output:02encoder/sampling_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
encoder/sampling_10/Shape_1Shape#encoder/dense_20/Relu:activations:0*
T0*
_output_shapes
::э╧s
)encoder/sampling_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+encoder/sampling_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+encoder/sampling_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#encoder/sampling_10/strided_slice_1StridedSlice$encoder/sampling_10/Shape_1:output:02encoder/sampling_10/strided_slice_1/stack:output:04encoder/sampling_10/strided_slice_1/stack_1:output:04encoder/sampling_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╖
'encoder/sampling_10/random_normal/shapePack*encoder/sampling_10/strided_slice:output:0,encoder/sampling_10/strided_slice_1:output:0*
N*
T0*
_output_shapes
:k
&encoder/sampling_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    m
(encoder/sampling_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▐
6encoder/sampling_10/random_normal/RandomStandardNormalRandomStandardNormal0encoder/sampling_10/random_normal/shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2▀лХ*
seed▒ х)╥
%encoder/sampling_10/random_normal/mulMul?encoder/sampling_10/random_normal/RandomStandardNormal:output:01encoder/sampling_10/random_normal/stddev:output:0*
T0*'
_output_shapes
:         @╕
!encoder/sampling_10/random_normalAddV2)encoder/sampling_10/random_normal/mul:z:0/encoder/sampling_10/random_normal/mean:output:0*
T0*'
_output_shapes
:         @^
encoder/sampling_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Щ
encoder/sampling_10/mulMul"encoder/sampling_10/mul/x:output:0#encoder/dense_21/Relu:activations:0*
T0*'
_output_shapes
:         @m
encoder/sampling_10/ExpExpencoder/sampling_10/mul:z:0*
T0*'
_output_shapes
:         @Ц
encoder/sampling_10/mul_1Mulencoder/sampling_10/Exp:y:0%encoder/sampling_10/random_normal:z:0*
T0*'
_output_shapes
:         @Ц
encoder/sampling_10/addAddV2#encoder/dense_20/Relu:activations:0encoder/sampling_10/mul_1:z:0*
T0*'
_output_shapes
:         @r
IdentityIdentity#encoder/dense_20/Relu:activations:0^NoOp*
T0*'
_output_shapes
:         @t

Identity_1Identity#encoder/dense_21/Relu:activations:0^NoOp*
T0*'
_output_shapes
:         @l

Identity_2Identityencoder/sampling_10/add:z:0^NoOp*
T0*'
_output_shapes
:         @Ё
NoOpNoOp(^encoder/dense_20/BiasAdd/ReadVariableOp'^encoder/dense_20/MatMul/ReadVariableOp(^encoder/dense_21/BiasAdd/ReadVariableOp'^encoder/dense_21/MatMul/ReadVariableOpP^encoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_40/batch_normalization_74/ReadVariableOpA^encoder/encoder_block_40/batch_normalization_74/ReadVariableOp_1:^encoder/encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp9^encoder/encoder_block_40/conv2d_86/Conv2D/ReadVariableOpP^encoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_41/batch_normalization_75/ReadVariableOpA^encoder/encoder_block_41/batch_normalization_75/ReadVariableOp_1:^encoder/encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp9^encoder/encoder_block_41/conv2d_87/Conv2D/ReadVariableOpP^encoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_42/batch_normalization_76/ReadVariableOpA^encoder/encoder_block_42/batch_normalization_76/ReadVariableOp_1:^encoder/encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp9^encoder/encoder_block_42/conv2d_88/Conv2D/ReadVariableOpP^encoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_43/batch_normalization_77/ReadVariableOpA^encoder/encoder_block_43/batch_normalization_77/ReadVariableOp_1:^encoder/encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp9^encoder/encoder_block_43/conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'encoder/dense_20/BiasAdd/ReadVariableOp'encoder/dense_20/BiasAdd/ReadVariableOp2P
&encoder/dense_20/MatMul/ReadVariableOp&encoder/dense_20/MatMul/ReadVariableOp2R
'encoder/dense_21/BiasAdd/ReadVariableOp'encoder/dense_21/BiasAdd/ReadVariableOp2P
&encoder/dense_21/MatMul/ReadVariableOp&encoder/dense_21/MatMul/ReadVariableOp2ж
Qencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12в
Oencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp2Д
@encoder/encoder_block_40/batch_normalization_74/ReadVariableOp_1@encoder/encoder_block_40/batch_normalization_74/ReadVariableOp_12А
>encoder/encoder_block_40/batch_normalization_74/ReadVariableOp>encoder/encoder_block_40/batch_normalization_74/ReadVariableOp2v
9encoder/encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp9encoder/encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_40/conv2d_86/Conv2D/ReadVariableOp8encoder/encoder_block_40/conv2d_86/Conv2D/ReadVariableOp2ж
Qencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12в
Oencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp2Д
@encoder/encoder_block_41/batch_normalization_75/ReadVariableOp_1@encoder/encoder_block_41/batch_normalization_75/ReadVariableOp_12А
>encoder/encoder_block_41/batch_normalization_75/ReadVariableOp>encoder/encoder_block_41/batch_normalization_75/ReadVariableOp2v
9encoder/encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp9encoder/encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_41/conv2d_87/Conv2D/ReadVariableOp8encoder/encoder_block_41/conv2d_87/Conv2D/ReadVariableOp2ж
Qencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12в
Oencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp2Д
@encoder/encoder_block_42/batch_normalization_76/ReadVariableOp_1@encoder/encoder_block_42/batch_normalization_76/ReadVariableOp_12А
>encoder/encoder_block_42/batch_normalization_76/ReadVariableOp>encoder/encoder_block_42/batch_normalization_76/ReadVariableOp2v
9encoder/encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp9encoder/encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_42/conv2d_88/Conv2D/ReadVariableOp8encoder/encoder_block_42/conv2d_88/Conv2D/ReadVariableOp2ж
Qencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12в
Oencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp2Д
@encoder/encoder_block_43/batch_normalization_77/ReadVariableOp_1@encoder/encoder_block_43/batch_normalization_77/ReadVariableOp_12А
>encoder/encoder_block_43/batch_normalization_77/ReadVariableOp>encoder/encoder_block_43/batch_normalization_77/ReadVariableOp2v
9encoder/encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp9encoder/encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_43/conv2d_89/Conv2D/ReadVariableOp8encoder/encoder_block_43/conv2d_89/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:         АА
!
_user_specified_name	input_1
Ў
Я
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617477
input_tensorD
(conv2d_87_conv2d_readvariableop_resource:АА8
)conv2d_87_biasadd_readvariableop_resource:	А=
.batch_normalization_75_readvariableop_resource:	А?
0batch_normalization_75_readvariableop_1_resource:	АN
?batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв6batch_normalization_75/FusedBatchNormV3/ReadVariableOpв8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_75/ReadVariableOpв'batch_normalization_75/ReadVariableOp_1в conv2d_87/BiasAdd/ReadVariableOpвconv2d_87/Conv2D/ReadVariableOpТ
conv2d_87/Conv2D/ReadVariableOpReadVariableOp(conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_87/Conv2DConv2Dinput_tensor'conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
З
 conv2d_87/BiasAdd/ReadVariableOpReadVariableOp)conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_87/BiasAddBiasAddconv2d_87/Conv2D:output:0(conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@Аm
conv2d_87/ReluReluconv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@Ап
max_pooling2d_41/MaxPoolMaxPoolconv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_75/ReadVariableOpReadVariableOp.batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_75/ReadVariableOp_1ReadVariableOp0batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╔
'batch_normalization_75/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_41/MaxPool:output:0-batch_normalization_75/ReadVariableOp:value:0/batch_normalization_75/ReadVariableOp_1:value:0>batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( Г
IdentityIdentity+batch_normalization_75/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           А╤
NoOpNoOp7^batch_normalization_75/FusedBatchNormV3/ReadVariableOp9^batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_75/ReadVariableOp(^batch_normalization_75/ReadVariableOp_1!^conv2d_87/BiasAdd/ReadVariableOp ^conv2d_87/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@А: : : : : : 2t
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_18batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_75/FusedBatchNormV3/ReadVariableOp6batch_normalization_75/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_75/ReadVariableOp_1'batch_normalization_75/ReadVariableOp_12N
%batch_normalization_75/ReadVariableOp%batch_normalization_75/ReadVariableOp2D
 conv2d_87/BiasAdd/ReadVariableOp conv2d_87/BiasAdd/ReadVariableOp2B
conv2d_87/Conv2D/ReadVariableOpconv2d_87/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@А
&
_user_specified_nameinput_tensor
а'
ё
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617451
input_tensorD
(conv2d_87_conv2d_readvariableop_resource:АА8
)conv2d_87_biasadd_readvariableop_resource:	А=
.batch_normalization_75_readvariableop_resource:	А?
0batch_normalization_75_readvariableop_1_resource:	АN
?batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	АP
Abatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв%batch_normalization_75/AssignNewValueв'batch_normalization_75/AssignNewValue_1в6batch_normalization_75/FusedBatchNormV3/ReadVariableOpв8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_75/ReadVariableOpв'batch_normalization_75/ReadVariableOp_1в conv2d_87/BiasAdd/ReadVariableOpвconv2d_87/Conv2D/ReadVariableOpТ
conv2d_87/Conv2D/ReadVariableOpReadVariableOp(conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0┤
conv2d_87/Conv2DConv2Dinput_tensor'conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
З
 conv2d_87/BiasAdd/ReadVariableOpReadVariableOp)conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ь
conv2d_87/BiasAddBiasAddconv2d_87/Conv2D:output:0(conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@Аm
conv2d_87/ReluReluconv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@Ап
max_pooling2d_41/MaxPoolMaxPoolconv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
С
%batch_normalization_75/ReadVariableOpReadVariableOp.batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0Х
'batch_normalization_75/ReadVariableOp_1ReadVariableOp0batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0│
6batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╫
'batch_normalization_75/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_41/MaxPool:output:0-batch_normalization_75/ReadVariableOp:value:0/batch_normalization_75/ReadVariableOp_1:value:0>batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<в
%batch_normalization_75/AssignNewValueAssignVariableOp?batch_normalization_75_fusedbatchnormv3_readvariableop_resource4batch_normalization_75/FusedBatchNormV3:batch_mean:07^batch_normalization_75/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(м
'batch_normalization_75/AssignNewValue_1AssignVariableOpAbatch_normalization_75_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_75/FusedBatchNormV3:batch_variance:09^batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Г
IdentityIdentity+batch_normalization_75/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           Аг
NoOpNoOp&^batch_normalization_75/AssignNewValue(^batch_normalization_75/AssignNewValue_17^batch_normalization_75/FusedBatchNormV3/ReadVariableOp9^batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_75/ReadVariableOp(^batch_normalization_75/ReadVariableOp_1!^conv2d_87/BiasAdd/ReadVariableOp ^conv2d_87/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@А: : : : : : 2R
'batch_normalization_75/AssignNewValue_1'batch_normalization_75/AssignNewValue_12N
%batch_normalization_75/AssignNewValue%batch_normalization_75/AssignNewValue2t
8batch_normalization_75/FusedBatchNormV3/ReadVariableOp_18batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_75/FusedBatchNormV3/ReadVariableOp6batch_normalization_75/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_75/ReadVariableOp_1'batch_normalization_75/ReadVariableOp_12N
%batch_normalization_75/ReadVariableOp%batch_normalization_75/ReadVariableOp2D
 conv2d_87/BiasAdd/ReadVariableOp conv2d_87/BiasAdd/ReadVariableOp2B
conv2d_87/Conv2D/ReadVariableOpconv2d_87/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@А
&
_user_specified_nameinput_tensor
Х
i
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2615639

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
а	
╫
8__inference_batch_normalization_74_layer_call_fn_2617755

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2615588К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
р
Щ
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617649
input_tensorC
(conv2d_89_conv2d_readvariableop_resource:А@7
)conv2d_89_biasadd_readvariableop_resource:@<
.batch_normalization_77_readvariableop_resource:@>
0batch_normalization_77_readvariableop_1_resource:@M
?batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@
identityИв6batch_normalization_77/FusedBatchNormV3/ReadVariableOpв8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_77/ReadVariableOpв'batch_normalization_77/ReadVariableOp_1в conv2d_89/BiasAdd/ReadVariableOpвconv2d_89/Conv2D/ReadVariableOpС
conv2d_89/Conv2D/ReadVariableOpReadVariableOp(conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0│
conv2d_89/Conv2DConv2Dinput_tensor'conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
Ж
 conv2d_89/BiasAdd/ReadVariableOpReadVariableOp)conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ы
conv2d_89/BiasAddBiasAddconv2d_89/Conv2D:output:0(conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @l
conv2d_89/ReluReluconv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @о
max_pooling2d_43/MaxPoolMaxPoolconv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Р
%batch_normalization_77/ReadVariableOpReadVariableOp.batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0Ф
'batch_normalization_77/ReadVariableOp_1ReadVariableOp0batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0─
'batch_normalization_77/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_43/MaxPool:output:0-batch_normalization_77/ReadVariableOp:value:0/batch_normalization_77/ReadVariableOp_1:value:0>batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( В
IdentityIdentity+batch_normalization_77/FusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:         @╤
NoOpNoOp7^batch_normalization_77/FusedBatchNormV3/ReadVariableOp9^batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_77/ReadVariableOp(^batch_normalization_77/ReadVariableOp_1!^conv2d_89/BiasAdd/ReadVariableOp ^conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         А: : : : : : 2t
8batch_normalization_77/FusedBatchNormV3/ReadVariableOp_18batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_77/FusedBatchNormV3/ReadVariableOp6batch_normalization_77/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_77/ReadVariableOp_1'batch_normalization_77/ReadVariableOp_12N
%batch_normalization_77/ReadVariableOp%batch_normalization_77/ReadVariableOp2D
 conv2d_89/BiasAdd/ReadVariableOp conv2d_89/BiasAdd/ReadVariableOp2B
conv2d_89/Conv2D/ReadVariableOpconv2d_89/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         А
&
_user_specified_nameinput_tensor
▐
в
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617876

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
░7
░
D__inference_encoder_layer_call_and_return_conditional_losses_2616480
tensor_input3
encoder_block_40_2616413:А'
encoder_block_40_2616415:	А'
encoder_block_40_2616417:	А'
encoder_block_40_2616419:	А'
encoder_block_40_2616421:	А'
encoder_block_40_2616423:	А4
encoder_block_41_2616426:АА'
encoder_block_41_2616428:	А'
encoder_block_41_2616430:	А'
encoder_block_41_2616432:	А'
encoder_block_41_2616434:	А'
encoder_block_41_2616436:	А4
encoder_block_42_2616439:АА'
encoder_block_42_2616441:	А'
encoder_block_42_2616443:	А'
encoder_block_42_2616445:	А'
encoder_block_42_2616447:	А'
encoder_block_42_2616449:	А3
encoder_block_43_2616452:А@&
encoder_block_43_2616454:@&
encoder_block_43_2616456:@&
encoder_block_43_2616458:@&
encoder_block_43_2616460:@&
encoder_block_43_2616462:@#
dense_20_2616466:	А @
dense_20_2616468:@#
dense_21_2616471:	А @
dense_21_2616473:@
identity

identity_1

identity_2Ив dense_20/StatefulPartitionedCallв dense_21/StatefulPartitionedCallв(encoder_block_40/StatefulPartitionedCallв(encoder_block_41/StatefulPartitionedCallв(encoder_block_42/StatefulPartitionedCallв(encoder_block_43/StatefulPartitionedCallв#sampling_10/StatefulPartitionedCallЧ
(encoder_block_40/StatefulPartitionedCallStatefulPartitionedCalltensor_inputencoder_block_40_2616413encoder_block_40_2616415encoder_block_40_2616417encoder_block_40_2616419encoder_block_40_2616421encoder_block_40_2616423*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2616126╝
(encoder_block_41/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_40/StatefulPartitionedCall:output:0encoder_block_41_2616426encoder_block_41_2616428encoder_block_41_2616430encoder_block_41_2616432encoder_block_41_2616434encoder_block_41_2616436*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2616165╝
(encoder_block_42/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_41/StatefulPartitionedCall:output:0encoder_block_42_2616439encoder_block_42_2616441encoder_block_42_2616443encoder_block_42_2616445encoder_block_42_2616447encoder_block_42_2616449*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2616204╗
(encoder_block_43/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_42/StatefulPartitionedCall:output:0encoder_block_43_2616452encoder_block_43_2616454encoder_block_43_2616456encoder_block_43_2616458encoder_block_43_2616460encoder_block_43_2616462*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8В *V
fQRO
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2616243ю
flatten_10/PartitionedCallPartitionedCall1encoder_block_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *P
fKRI
G__inference_flatten_10_layer_call_and_return_conditional_losses_2616031Х
 dense_20/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_20_2616466dense_20_2616468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_20_layer_call_and_return_conditional_losses_2616044Х
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_21_2616471dense_21_2616473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *N
fIRG
E__inference_dense_21_layer_call_and_return_conditional_losses_2616061г
#sampling_10/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8В *Q
fLRJ
H__inference_sampling_10_layer_call_and_return_conditional_losses_2616093x
IdentityIdentity)dense_20/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @z

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @}

Identity_2Identity,sampling_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @▐
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall)^encoder_block_40/StatefulPartitionedCall)^encoder_block_41/StatefulPartitionedCall)^encoder_block_42/StatefulPartitionedCall)^encoder_block_43/StatefulPartitionedCall$^sampling_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2T
(encoder_block_40/StatefulPartitionedCall(encoder_block_40/StatefulPartitionedCall2T
(encoder_block_41/StatefulPartitionedCall(encoder_block_41/StatefulPartitionedCall2T
(encoder_block_42/StatefulPartitionedCall(encoder_block_42/StatefulPartitionedCall2T
(encoder_block_43/StatefulPartitionedCall(encoder_block_43/StatefulPartitionedCall2J
#sampling_10/StatefulPartitionedCall#sampling_10/StatefulPartitionedCall:_ [
1
_output_shapes
:         АА
&
_user_specified_nametensor_input
▐
в
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2615758

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
в	
╫
8__inference_batch_normalization_74_layer_call_fn_2617768

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8В *\
fWRU
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2615606К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
з
w
H__inference_sampling_10_layer_call_and_return_conditional_losses_2617732
inputs_0
inputs_1
identityИK
ShapeShapeinputs_0*
T0*
_output_shapes
::э╧]
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
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_0*
T0*
_output_shapes
::э╧_
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
valueB:█
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
 *  А?╢
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2аоИ*
seed▒ х)Ц
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         @|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         @J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?V
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:         @E
ExpExpmul:z:0*
T0*'
_output_shapes
:         @Z
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:         @S
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:         @O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         @:         @:QM
'
_output_shapes
:         @
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:         @
"
_user_specified_name
inputs_0
▐
в
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617948

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Х
i
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2615563

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
Х
i
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2615715

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
Г╛
╘
D__inference_encoder_layer_call_and_return_conditional_losses_2617305
tensor_inputT
9encoder_block_40_conv2d_86_conv2d_readvariableop_resource:АI
:encoder_block_40_conv2d_86_biasadd_readvariableop_resource:	АN
?encoder_block_40_batch_normalization_74_readvariableop_resource:	АP
Aencoder_block_40_batch_normalization_74_readvariableop_1_resource:	А_
Pencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resource:	Аa
Rencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resource:	АU
9encoder_block_41_conv2d_87_conv2d_readvariableop_resource:ААI
:encoder_block_41_conv2d_87_biasadd_readvariableop_resource:	АN
?encoder_block_41_batch_normalization_75_readvariableop_resource:	АP
Aencoder_block_41_batch_normalization_75_readvariableop_1_resource:	А_
Pencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resource:	Аa
Rencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resource:	АU
9encoder_block_42_conv2d_88_conv2d_readvariableop_resource:ААI
:encoder_block_42_conv2d_88_biasadd_readvariableop_resource:	АN
?encoder_block_42_batch_normalization_76_readvariableop_resource:	АP
Aencoder_block_42_batch_normalization_76_readvariableop_1_resource:	А_
Pencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resource:	Аa
Rencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource:	АT
9encoder_block_43_conv2d_89_conv2d_readvariableop_resource:А@H
:encoder_block_43_conv2d_89_biasadd_readvariableop_resource:@M
?encoder_block_43_batch_normalization_77_readvariableop_resource:@O
Aencoder_block_43_batch_normalization_77_readvariableop_1_resource:@^
Pencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resource:@`
Rencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource:@:
'dense_20_matmul_readvariableop_resource:	А @6
(dense_20_biasadd_readvariableop_resource:@:
'dense_21_matmul_readvariableop_resource:	А @6
(dense_21_biasadd_readvariableop_resource:@
identity

identity_1

identity_2Ивdense_20/BiasAdd/ReadVariableOpвdense_20/MatMul/ReadVariableOpвdense_21/BiasAdd/ReadVariableOpвdense_21/MatMul/ReadVariableOpвGencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpвIencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_40/batch_normalization_74/ReadVariableOpв8encoder_block_40/batch_normalization_74/ReadVariableOp_1в1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOpв0encoder_block_40/conv2d_86/Conv2D/ReadVariableOpвGencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpвIencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_41/batch_normalization_75/ReadVariableOpв8encoder_block_41/batch_normalization_75/ReadVariableOp_1в1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOpв0encoder_block_41/conv2d_87/Conv2D/ReadVariableOpвGencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpвIencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_42/batch_normalization_76/ReadVariableOpв8encoder_block_42/batch_normalization_76/ReadVariableOp_1в1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOpв0encoder_block_42/conv2d_88/Conv2D/ReadVariableOpвGencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpвIencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1в6encoder_block_43/batch_normalization_77/ReadVariableOpв8encoder_block_43/batch_normalization_77/ReadVariableOp_1в1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOpв0encoder_block_43/conv2d_89/Conv2D/ReadVariableOp│
0encoder_block_40/conv2d_86/Conv2D/ReadVariableOpReadVariableOp9encoder_block_40_conv2d_86_conv2d_readvariableop_resource*'
_output_shapes
:А*
dtype0╪
!encoder_block_40/conv2d_86/Conv2DConv2Dtensor_input8encoder_block_40/conv2d_86/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ААА*
paddingSAME*
strides
й
1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_40_conv2d_86_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╤
"encoder_block_40/conv2d_86/BiasAddBiasAdd*encoder_block_40/conv2d_86/Conv2D:output:09encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         АААС
encoder_block_40/conv2d_86/ReluRelu+encoder_block_40/conv2d_86/BiasAdd:output:0*
T0*2
_output_shapes 
:         ААА╤
)encoder_block_40/max_pooling2d_40/MaxPoolMaxPool-encoder_block_40/conv2d_86/Relu:activations:0*0
_output_shapes
:         @@А*
ksize
*
paddingVALID*
strides
│
6encoder_block_40/batch_normalization_74/ReadVariableOpReadVariableOp?encoder_block_40_batch_normalization_74_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8encoder_block_40/batch_normalization_74/ReadVariableOp_1ReadVariableOpAencoder_block_40_batch_normalization_74_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╒
Gencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┘
Iencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_40_batch_normalization_74_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0п
8encoder_block_40/batch_normalization_74/FusedBatchNormV3FusedBatchNormV32encoder_block_40/max_pooling2d_40/MaxPool:output:0>encoder_block_40/batch_normalization_74/ReadVariableOp:value:0@encoder_block_40/batch_normalization_74/ReadVariableOp_1:value:0Oencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@А:А:А:А:А:*
epsilon%oГ:*
is_training( ┤
0encoder_block_41/conv2d_87/Conv2D/ReadVariableOpReadVariableOp9encoder_block_41_conv2d_87_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
!encoder_block_41/conv2d_87/Conv2DConv2D<encoder_block_40/batch_normalization_74/FusedBatchNormV3:y:08encoder_block_41/conv2d_87/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@А*
paddingSAME*
strides
й
1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_41_conv2d_87_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╧
"encoder_block_41/conv2d_87/BiasAddBiasAdd*encoder_block_41/conv2d_87/Conv2D:output:09encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@АП
encoder_block_41/conv2d_87/ReluRelu+encoder_block_41/conv2d_87/BiasAdd:output:0*
T0*0
_output_shapes
:         @@А╤
)encoder_block_41/max_pooling2d_41/MaxPoolMaxPool-encoder_block_41/conv2d_87/Relu:activations:0*0
_output_shapes
:           А*
ksize
*
paddingVALID*
strides
│
6encoder_block_41/batch_normalization_75/ReadVariableOpReadVariableOp?encoder_block_41_batch_normalization_75_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8encoder_block_41/batch_normalization_75/ReadVariableOp_1ReadVariableOpAencoder_block_41_batch_normalization_75_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╒
Gencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┘
Iencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_41_batch_normalization_75_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0п
8encoder_block_41/batch_normalization_75/FusedBatchNormV3FusedBatchNormV32encoder_block_41/max_pooling2d_41/MaxPool:output:0>encoder_block_41/batch_normalization_75/ReadVariableOp:value:0@encoder_block_41/batch_normalization_75/ReadVariableOp_1:value:0Oencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           А:А:А:А:А:*
epsilon%oГ:*
is_training( ┤
0encoder_block_42/conv2d_88/Conv2D/ReadVariableOpReadVariableOp9encoder_block_42_conv2d_88_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
!encoder_block_42/conv2d_88/Conv2DConv2D<encoder_block_41/batch_normalization_75/FusedBatchNormV3:y:08encoder_block_42/conv2d_88/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           А*
paddingSAME*
strides
й
1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_42_conv2d_88_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╧
"encoder_block_42/conv2d_88/BiasAddBiasAdd*encoder_block_42/conv2d_88/Conv2D:output:09encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           АП
encoder_block_42/conv2d_88/ReluRelu+encoder_block_42/conv2d_88/BiasAdd:output:0*
T0*0
_output_shapes
:           А╤
)encoder_block_42/max_pooling2d_42/MaxPoolMaxPool-encoder_block_42/conv2d_88/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
│
6encoder_block_42/batch_normalization_76/ReadVariableOpReadVariableOp?encoder_block_42_batch_normalization_76_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
8encoder_block_42/batch_normalization_76/ReadVariableOp_1ReadVariableOpAencoder_block_42_batch_normalization_76_readvariableop_1_resource*
_output_shapes	
:А*
dtype0╒
Gencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0┘
Iencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_42_batch_normalization_76_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0п
8encoder_block_42/batch_normalization_76/FusedBatchNormV3FusedBatchNormV32encoder_block_42/max_pooling2d_42/MaxPool:output:0>encoder_block_42/batch_normalization_76/ReadVariableOp:value:0@encoder_block_42/batch_normalization_76/ReadVariableOp_1:value:0Oencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         А:А:А:А:А:*
epsilon%oГ:*
is_training( │
0encoder_block_43/conv2d_89/Conv2D/ReadVariableOpReadVariableOp9encoder_block_43_conv2d_89_conv2d_readvariableop_resource*'
_output_shapes
:А@*
dtype0Е
!encoder_block_43/conv2d_89/Conv2DConv2D<encoder_block_42/batch_normalization_76/FusedBatchNormV3:y:08encoder_block_43/conv2d_89/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingSAME*
strides
и
1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_43_conv2d_89_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╬
"encoder_block_43/conv2d_89/BiasAddBiasAdd*encoder_block_43/conv2d_89/Conv2D:output:09encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
encoder_block_43/conv2d_89/ReluRelu+encoder_block_43/conv2d_89/BiasAdd:output:0*
T0*/
_output_shapes
:         @╨
)encoder_block_43/max_pooling2d_43/MaxPoolMaxPool-encoder_block_43/conv2d_89/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
▓
6encoder_block_43/batch_normalization_77/ReadVariableOpReadVariableOp?encoder_block_43_batch_normalization_77_readvariableop_resource*
_output_shapes
:@*
dtype0╢
8encoder_block_43/batch_normalization_77/ReadVariableOp_1ReadVariableOpAencoder_block_43_batch_normalization_77_readvariableop_1_resource*
_output_shapes
:@*
dtype0╘
Gencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0╪
Iencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_43_batch_normalization_77_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0к
8encoder_block_43/batch_normalization_77/FusedBatchNormV3FusedBatchNormV32encoder_block_43/max_pooling2d_43/MaxPool:output:0>encoder_block_43/batch_normalization_77/ReadVariableOp:value:0@encoder_block_43/batch_normalization_77/ReadVariableOp_1:value:0Oencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( a
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"       й
flatten_10/ReshapeReshape<encoder_block_43/batch_normalization_77/FusedBatchNormV3:y:0flatten_10/Const:output:0*
T0*(
_output_shapes
:         А З
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0Р
dense_20/MatMulMatMulflatten_10/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Д
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:         @З
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	А @*
dtype0Р
dense_21/MatMulMatMulflatten_10/Reshape:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Д
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0С
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @b
dense_21/ReluReludense_21/BiasAdd:output:0*
T0*'
_output_shapes
:         @j
sampling_10/ShapeShapedense_20/Relu:activations:0*
T0*
_output_shapes
::э╧i
sampling_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!sampling_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!sampling_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
sampling_10/strided_sliceStridedSlicesampling_10/Shape:output:0(sampling_10/strided_slice/stack:output:0*sampling_10/strided_slice/stack_1:output:0*sampling_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
sampling_10/Shape_1Shapedense_20/Relu:activations:0*
T0*
_output_shapes
::э╧k
!sampling_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#sampling_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
sampling_10/strided_slice_1StridedSlicesampling_10/Shape_1:output:0*sampling_10/strided_slice_1/stack:output:0,sampling_10/strided_slice_1/stack_1:output:0,sampling_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЯ
sampling_10/random_normal/shapePack"sampling_10/strided_slice:output:0$sampling_10/strided_slice_1:output:0*
N*
T0*
_output_shapes
:c
sampling_10/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    e
 sampling_10/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╬
.sampling_10/random_normal/RandomStandardNormalRandomStandardNormal(sampling_10/random_normal/shape:output:0*
T0*'
_output_shapes
:         @*
dtype0*
seed2Ч░ж*
seed▒ х)║
sampling_10/random_normal/mulMul7sampling_10/random_normal/RandomStandardNormal:output:0)sampling_10/random_normal/stddev:output:0*
T0*'
_output_shapes
:         @а
sampling_10/random_normalAddV2!sampling_10/random_normal/mul:z:0'sampling_10/random_normal/mean:output:0*
T0*'
_output_shapes
:         @V
sampling_10/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Б
sampling_10/mulMulsampling_10/mul/x:output:0dense_21/Relu:activations:0*
T0*'
_output_shapes
:         @]
sampling_10/ExpExpsampling_10/mul:z:0*
T0*'
_output_shapes
:         @~
sampling_10/mul_1Mulsampling_10/Exp:y:0sampling_10/random_normal:z:0*
T0*'
_output_shapes
:         @~
sampling_10/addAddV2dense_20/Relu:activations:0sampling_10/mul_1:z:0*
T0*'
_output_shapes
:         @j
IdentityIdentitydense_20/Relu:activations:0^NoOp*
T0*'
_output_shapes
:         @l

Identity_1Identitydense_21/Relu:activations:0^NoOp*
T0*'
_output_shapes
:         @d

Identity_2Identitysampling_10/add:z:0^NoOp*
T0*'
_output_shapes
:         @Р
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOpH^encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpJ^encoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_17^encoder_block_40/batch_normalization_74/ReadVariableOp9^encoder_block_40/batch_normalization_74/ReadVariableOp_12^encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp1^encoder_block_40/conv2d_86/Conv2D/ReadVariableOpH^encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpJ^encoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_17^encoder_block_41/batch_normalization_75/ReadVariableOp9^encoder_block_41/batch_normalization_75/ReadVariableOp_12^encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp1^encoder_block_41/conv2d_87/Conv2D/ReadVariableOpH^encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpJ^encoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_17^encoder_block_42/batch_normalization_76/ReadVariableOp9^encoder_block_42/batch_normalization_76/ReadVariableOp_12^encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp1^encoder_block_42/conv2d_88/Conv2D/ReadVariableOpH^encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpJ^encoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_17^encoder_block_43/batch_normalization_77/ReadVariableOp9^encoder_block_43/batch_normalization_77/ReadVariableOp_12^encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp1^encoder_block_43/conv2d_89/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2Ц
Iencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOpGencoder_block_40/batch_normalization_74/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_40/batch_normalization_74/ReadVariableOp_18encoder_block_40/batch_normalization_74/ReadVariableOp_12p
6encoder_block_40/batch_normalization_74/ReadVariableOp6encoder_block_40/batch_normalization_74/ReadVariableOp2f
1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp1encoder_block_40/conv2d_86/BiasAdd/ReadVariableOp2d
0encoder_block_40/conv2d_86/Conv2D/ReadVariableOp0encoder_block_40/conv2d_86/Conv2D/ReadVariableOp2Ц
Iencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOpGencoder_block_41/batch_normalization_75/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_41/batch_normalization_75/ReadVariableOp_18encoder_block_41/batch_normalization_75/ReadVariableOp_12p
6encoder_block_41/batch_normalization_75/ReadVariableOp6encoder_block_41/batch_normalization_75/ReadVariableOp2f
1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp1encoder_block_41/conv2d_87/BiasAdd/ReadVariableOp2d
0encoder_block_41/conv2d_87/Conv2D/ReadVariableOp0encoder_block_41/conv2d_87/Conv2D/ReadVariableOp2Ц
Iencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOpGencoder_block_42/batch_normalization_76/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_42/batch_normalization_76/ReadVariableOp_18encoder_block_42/batch_normalization_76/ReadVariableOp_12p
6encoder_block_42/batch_normalization_76/ReadVariableOp6encoder_block_42/batch_normalization_76/ReadVariableOp2f
1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp1encoder_block_42/conv2d_88/BiasAdd/ReadVariableOp2d
0encoder_block_42/conv2d_88/Conv2D/ReadVariableOp0encoder_block_42/conv2d_88/Conv2D/ReadVariableOp2Ц
Iencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp_12Т
Gencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOpGencoder_block_43/batch_normalization_77/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_43/batch_normalization_77/ReadVariableOp_18encoder_block_43/batch_normalization_77/ReadVariableOp_12p
6encoder_block_43/batch_normalization_77/ReadVariableOp6encoder_block_43/batch_normalization_77/ReadVariableOp2f
1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp1encoder_block_43/conv2d_89/BiasAdd/ReadVariableOp2d
0encoder_block_43/conv2d_89/Conv2D/ReadVariableOp0encoder_block_43/conv2d_89/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         АА
&
_user_specified_nametensor_input
╧
Э
)__inference_encoder_layer_call_fn_2616543
input_1"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А&

unknown_11:АА

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А%

unknown_17:А@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	А @

unknown_24:@

unknown_25:	А @

unknown_26:@
identity

identity_1

identity_2ИвStatefulPartitionedCallъ
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
 *M
_output_shapes;
9:         @:         @:         @*>
_read_only_resource_inputs 
	
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_2616480o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         АА
!
_user_specified_name	input_1
ер
б 
 __inference__traced_save_2618213
file_prefix[
@read_disablecopyonread_encoder_encoder_block_40_conv2d_86_kernel:АO
@read_1_disablecopyonread_encoder_encoder_block_40_conv2d_86_bias:	А]
Nread_2_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_gamma:	А\
Mread_3_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_beta:	Аc
Tread_4_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_moving_mean:	Аg
Xread_5_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_moving_variance:	А^
Bread_6_disablecopyonread_encoder_encoder_block_41_conv2d_87_kernel:ААO
@read_7_disablecopyonread_encoder_encoder_block_41_conv2d_87_bias:	А]
Nread_8_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_gamma:	А\
Mread_9_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_beta:	Аd
Uread_10_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_moving_mean:	Аh
Yread_11_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_moving_variance:	А_
Cread_12_disablecopyonread_encoder_encoder_block_42_conv2d_88_kernel:ААP
Aread_13_disablecopyonread_encoder_encoder_block_42_conv2d_88_bias:	А^
Oread_14_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_gamma:	А]
Nread_15_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_beta:	Аd
Uread_16_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_moving_mean:	Аh
Yread_17_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_moving_variance:	А^
Cread_18_disablecopyonread_encoder_encoder_block_43_conv2d_89_kernel:А@O
Aread_19_disablecopyonread_encoder_encoder_block_43_conv2d_89_bias:@]
Oread_20_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_gamma:@\
Nread_21_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_beta:@c
Uread_22_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_moving_mean:@g
Yread_23_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_moving_variance:@D
1read_24_disablecopyonread_encoder_dense_20_kernel:	А @=
/read_25_disablecopyonread_encoder_dense_20_bias:@D
1read_26_disablecopyonread_encoder_dense_21_kernel:	А @=
/read_27_disablecopyonread_encoder_dense_21_bias:@
savev2_const
identity_57ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Т
Read/DisableCopyOnReadDisableCopyOnRead@read_disablecopyonread_encoder_encoder_block_40_conv2d_86_kernel"/device:CPU:0*
_output_shapes
 ┼
Read/ReadVariableOpReadVariableOp@read_disablecopyonread_encoder_encoder_block_40_conv2d_86_kernel^Read/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:А*
dtype0r
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:Аj

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*'
_output_shapes
:АФ
Read_1/DisableCopyOnReadDisableCopyOnRead@read_1_disablecopyonread_encoder_encoder_block_40_conv2d_86_bias"/device:CPU:0*
_output_shapes
 ╜
Read_1/ReadVariableOpReadVariableOp@read_1_disablecopyonread_encoder_encoder_block_40_conv2d_86_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ав
Read_2/DisableCopyOnReadDisableCopyOnReadNread_2_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_2/ReadVariableOpReadVariableOpNread_2_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аб
Read_3/DisableCopyOnReadDisableCopyOnReadMread_3_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_beta"/device:CPU:0*
_output_shapes
 ╩
Read_3/ReadVariableOpReadVariableOpMread_3_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аи
Read_4/DisableCopyOnReadDisableCopyOnReadTread_4_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_moving_mean"/device:CPU:0*
_output_shapes
 ╤
Read_4/ReadVariableOpReadVariableOpTread_4_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:А`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ам
Read_5/DisableCopyOnReadDisableCopyOnReadXread_5_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_moving_variance"/device:CPU:0*
_output_shapes
 ╒
Read_5/ReadVariableOpReadVariableOpXread_5_disablecopyonread_encoder_encoder_block_40_batch_normalization_74_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЦ
Read_6/DisableCopyOnReadDisableCopyOnReadBread_6_disablecopyonread_encoder_encoder_block_41_conv2d_87_kernel"/device:CPU:0*
_output_shapes
 ╠
Read_6/ReadVariableOpReadVariableOpBread_6_disablecopyonread_encoder_encoder_block_41_conv2d_87_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0x
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААФ
Read_7/DisableCopyOnReadDisableCopyOnRead@read_7_disablecopyonread_encoder_encoder_block_41_conv2d_87_bias"/device:CPU:0*
_output_shapes
 ╜
Read_7/ReadVariableOpReadVariableOp@read_7_disablecopyonread_encoder_encoder_block_41_conv2d_87_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ав
Read_8/DisableCopyOnReadDisableCopyOnReadNread_8_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_8/ReadVariableOpReadVariableOpNread_8_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аб
Read_9/DisableCopyOnReadDisableCopyOnReadMread_9_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_beta"/device:CPU:0*
_output_shapes
 ╩
Read_9/ReadVariableOpReadVariableOpMread_9_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ак
Read_10/DisableCopyOnReadDisableCopyOnReadUread_10_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_moving_mean"/device:CPU:0*
_output_shapes
 ╘
Read_10/ReadVariableOpReadVariableOpUread_10_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ао
Read_11/DisableCopyOnReadDisableCopyOnReadYread_11_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_moving_variance"/device:CPU:0*
_output_shapes
 ╪
Read_11/ReadVariableOpReadVariableOpYread_11_disablecopyonread_encoder_encoder_block_41_batch_normalization_75_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:АШ
Read_12/DisableCopyOnReadDisableCopyOnReadCread_12_disablecopyonread_encoder_encoder_block_42_conv2d_88_kernel"/device:CPU:0*
_output_shapes
 ╧
Read_12/ReadVariableOpReadVariableOpCread_12_disablecopyonread_encoder_encoder_block_42_conv2d_88_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:АА*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ААo
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:ААЦ
Read_13/DisableCopyOnReadDisableCopyOnReadAread_13_disablecopyonread_encoder_encoder_block_42_conv2d_88_bias"/device:CPU:0*
_output_shapes
 └
Read_13/ReadVariableOpReadVariableOpAread_13_disablecopyonread_encoder_encoder_block_42_conv2d_88_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ад
Read_14/DisableCopyOnReadDisableCopyOnReadOread_14_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_gamma"/device:CPU:0*
_output_shapes
 ╬
Read_14/ReadVariableOpReadVariableOpOread_14_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аг
Read_15/DisableCopyOnReadDisableCopyOnReadNread_15_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_beta"/device:CPU:0*
_output_shapes
 ═
Read_15/ReadVariableOpReadVariableOpNread_15_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ак
Read_16/DisableCopyOnReadDisableCopyOnReadUread_16_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_moving_mean"/device:CPU:0*
_output_shapes
 ╘
Read_16/ReadVariableOpReadVariableOpUread_16_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ао
Read_17/DisableCopyOnReadDisableCopyOnReadYread_17_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_moving_variance"/device:CPU:0*
_output_shapes
 ╪
Read_17/ReadVariableOpReadVariableOpYread_17_disablecopyonread_encoder_encoder_block_42_batch_normalization_76_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:АШ
Read_18/DisableCopyOnReadDisableCopyOnReadCread_18_disablecopyonread_encoder_encoder_block_43_conv2d_89_kernel"/device:CPU:0*
_output_shapes
 ╬
Read_18/ReadVariableOpReadVariableOpCread_18_disablecopyonread_encoder_encoder_block_43_conv2d_89_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:А@*
dtype0x
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:А@n
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*'
_output_shapes
:А@Ц
Read_19/DisableCopyOnReadDisableCopyOnReadAread_19_disablecopyonread_encoder_encoder_block_43_conv2d_89_bias"/device:CPU:0*
_output_shapes
 ┐
Read_19/ReadVariableOpReadVariableOpAread_19_disablecopyonread_encoder_encoder_block_43_conv2d_89_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:@д
Read_20/DisableCopyOnReadDisableCopyOnReadOread_20_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_gamma"/device:CPU:0*
_output_shapes
 ═
Read_20/ReadVariableOpReadVariableOpOread_20_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
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
:@г
Read_21/DisableCopyOnReadDisableCopyOnReadNread_21_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_beta"/device:CPU:0*
_output_shapes
 ╠
Read_21/ReadVariableOpReadVariableOpNread_21_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
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
:@к
Read_22/DisableCopyOnReadDisableCopyOnReadUread_22_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_moving_mean"/device:CPU:0*
_output_shapes
 ╙
Read_22/ReadVariableOpReadVariableOpUread_22_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
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
:@о
Read_23/DisableCopyOnReadDisableCopyOnReadYread_23_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_moving_variance"/device:CPU:0*
_output_shapes
 ╫
Read_23/ReadVariableOpReadVariableOpYread_23_disablecopyonread_encoder_encoder_block_43_batch_normalization_77_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
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
:@Ж
Read_24/DisableCopyOnReadDisableCopyOnRead1read_24_disablecopyonread_encoder_dense_20_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_24/ReadVariableOpReadVariableOp1read_24_disablecopyonread_encoder_dense_20_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А @*
dtype0p
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А @f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	А @Д
Read_25/DisableCopyOnReadDisableCopyOnRead/read_25_disablecopyonread_encoder_dense_20_bias"/device:CPU:0*
_output_shapes
 н
Read_25/ReadVariableOpReadVariableOp/read_25_disablecopyonread_encoder_dense_20_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:@Ж
Read_26/DisableCopyOnReadDisableCopyOnRead1read_26_disablecopyonread_encoder_dense_21_kernel"/device:CPU:0*
_output_shapes
 ┤
Read_26/ReadVariableOpReadVariableOp1read_26_disablecopyonread_encoder_dense_21_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А @*
dtype0p
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А @f
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:	А @Д
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_encoder_dense_21_bias"/device:CPU:0*
_output_shapes
 н
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_encoder_dense_21_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@№	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*е	
valueЫ	BШ	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B ╫
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *+
dtypes!
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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
: П
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
Х
i
M__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_2615791

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
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
Ш
╞
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617786

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<╞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(╨
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А╘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╟
Э
)__inference_encoder_layer_call_fn_2616408
input_1"
unknown:А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
	unknown_3:	А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А
	unknown_7:	А
	unknown_8:	А
	unknown_9:	А

unknown_10:	А&

unknown_11:АА

unknown_12:	А

unknown_13:	А

unknown_14:	А

unknown_15:	А

unknown_16:	А%

unknown_17:А@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	А @

unknown_24:@

unknown_25:	А @

unknown_26:@
identity

identity_1

identity_2ИвStatefulPartitionedCallт
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
 *M
_output_shapes;
9:         @:         @:         @*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_2616345o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         АА: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         АА
!
_user_specified_name	input_1"є
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▒
serving_defaultЭ
E
input_1:
serving_default_input_1:0         АА<
output_10
StatefulPartitionedCall:0         @<
output_20
StatefulPartitionedCall:1         @<
output_30
StatefulPartitionedCall:2         @tensorflow/serving/predict:П▌
╗
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
Ў
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
╢
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
╩
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
╒
2trace_0
3trace_1
4trace_2
5trace_32ъ
)__inference_encoder_layer_call_fn_2616408
)__inference_encoder_layer_call_fn_2616543
)__inference_encoder_layer_call_fn_2616978
)__inference_encoder_layer_call_fn_2617043╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 z2trace_0z3trace_1z4trace_2z5trace_3
┴
6trace_0
7trace_1
8trace_2
9trace_32╓
D__inference_encoder_layer_call_and_return_conditional_losses_2616098
D__inference_encoder_layer_call_and_return_conditional_losses_2616272
D__inference_encoder_layer_call_and_return_conditional_losses_2617174
D__inference_encoder_layer_call_and_return_conditional_losses_2617305╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 z6trace_0z7trace_1z8trace_2z9trace_3
═B╩
"__inference__wrapped_model_2615557input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─
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
─
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
─
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
─
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
е
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
╗
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
е
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
D:BА2)encoder/encoder_block_40/conv2d_86/kernel
6:4А2'encoder/encoder_block_40/conv2d_86/bias
D:BА25encoder/encoder_block_40/batch_normalization_74/gamma
C:AА24encoder/encoder_block_40/batch_normalization_74/beta
L:JА (2;encoder/encoder_block_40/batch_normalization_74/moving_mean
P:NА (2?encoder/encoder_block_40/batch_normalization_74/moving_variance
E:CАА2)encoder/encoder_block_41/conv2d_87/kernel
6:4А2'encoder/encoder_block_41/conv2d_87/bias
D:BА25encoder/encoder_block_41/batch_normalization_75/gamma
C:AА24encoder/encoder_block_41/batch_normalization_75/beta
L:JА (2;encoder/encoder_block_41/batch_normalization_75/moving_mean
P:NА (2?encoder/encoder_block_41/batch_normalization_75/moving_variance
E:CАА2)encoder/encoder_block_42/conv2d_88/kernel
6:4А2'encoder/encoder_block_42/conv2d_88/bias
D:BА25encoder/encoder_block_42/batch_normalization_76/gamma
C:AА24encoder/encoder_block_42/batch_normalization_76/beta
L:JА (2;encoder/encoder_block_42/batch_normalization_76/moving_mean
P:NА (2?encoder/encoder_block_42/batch_normalization_76/moving_variance
D:BА@2)encoder/encoder_block_43/conv2d_89/kernel
5:3@2'encoder/encoder_block_43/conv2d_89/bias
C:A@25encoder/encoder_block_43/batch_normalization_77/gamma
B:@@24encoder/encoder_block_43/batch_normalization_77/beta
K:I@ (2;encoder/encoder_block_43/batch_normalization_77/moving_mean
O:M@ (2?encoder/encoder_block_43/batch_normalization_77/moving_variance
*:(	А @2encoder/dense_20/kernel
#:!@2encoder/dense_20/bias
*:(	А @2encoder/dense_21/kernel
#:!@2encoder/dense_21/bias
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
ўBЇ
)__inference_encoder_layer_call_fn_2616408input_1"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ўBЇ
)__inference_encoder_layer_call_fn_2616543input_1"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
№B∙
)__inference_encoder_layer_call_fn_2616978tensor_input"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
№B∙
)__inference_encoder_layer_call_fn_2617043tensor_input"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ТBП
D__inference_encoder_layer_call_and_return_conditional_losses_2616098input_1"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ТBП
D__inference_encoder_layer_call_and_return_conditional_losses_2616272input_1"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ЧBФ
D__inference_encoder_layer_call_and_return_conditional_losses_2617174tensor_input"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ЧBФ
D__inference_encoder_layer_call_and_return_conditional_losses_2617305tensor_input"╗
┤▓░
FullArgSpec
argsЪ
jtensor_input
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
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
н
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
╤
|trace_0
}trace_12Ъ
2__inference_encoder_block_40_layer_call_fn_2617322
2__inference_encoder_block_40_layer_call_fn_2617339п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z|trace_0z}trace_1
З
~trace_0
trace_12╨
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617365
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617391п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z~trace_0ztrace_1
ф
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses

kernel
bias
!Ж_jit_compiled_convolution_op"
_tf_keras_layer
л
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
	Уaxis
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
Фnon_trainable_variables
Хlayers
Цmetrics
 Чlayer_regularization_losses
Шlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
╒
Щtrace_0
Ъtrace_12Ъ
2__inference_encoder_block_41_layer_call_fn_2617408
2__inference_encoder_block_41_layer_call_fn_2617425п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЩtrace_0zЪtrace_1
Л
Ыtrace_0
Ьtrace_12╨
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617451
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617477п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0zЬtrace_1
ф
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses

kernel
bias
!г_jit_compiled_convolution_op"
_tf_keras_layer
л
д	variables
еtrainable_variables
жregularization_losses
з	keras_api
и__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses
	░axis
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
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
╒
╢trace_0
╖trace_12Ъ
2__inference_encoder_block_42_layer_call_fn_2617494
2__inference_encoder_block_42_layer_call_fn_2617511п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╢trace_0z╖trace_1
Л
╕trace_0
╣trace_12╨
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617537
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617563п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0z╣trace_1
ф
║	variables
╗trainable_variables
╝regularization_losses
╜	keras_api
╛__call__
+┐&call_and_return_all_conditional_losses

kernel
bias
!└_jit_compiled_convolution_op"
_tf_keras_layer
л
┴	variables
┬trainable_variables
├regularization_losses
─	keras_api
┼__call__
+╞&call_and_return_all_conditional_losses"
_tf_keras_layer
ё
╟	variables
╚trainable_variables
╔regularization_losses
╩	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses
	═axis
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
▓
╬non_trainable_variables
╧layers
╨metrics
 ╤layer_regularization_losses
╥layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
╒
╙trace_0
╘trace_12Ъ
2__inference_encoder_block_43_layer_call_fn_2617580
2__inference_encoder_block_43_layer_call_fn_2617597п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╙trace_0z╘trace_1
Л
╒trace_0
╓trace_12╨
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617623
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617649п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╒trace_0z╓trace_1
ф
╫	variables
╪trainable_variables
┘regularization_losses
┌	keras_api
█__call__
+▄&call_and_return_all_conditional_losses

#kernel
$bias
!▌_jit_compiled_convolution_op"
_tf_keras_layer
л
▐	variables
▀trainable_variables
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
▓
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
ш
Ёtrace_02╔
,__inference_flatten_10_layer_call_fn_2617654Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
Г
ёtrace_02ф
G__inference_flatten_10_layer_call_and_return_conditional_losses_2617660Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▓
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ц
ўtrace_02╟
*__inference_dense_20_layer_call_fn_2617669Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
Б
°trace_02т
E__inference_dense_20_layer_call_and_return_conditional_losses_2617680Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
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
▓
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ц
■trace_02╟
*__inference_dense_21_layer_call_fn_2617689Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
Б
 trace_02т
E__inference_dense_21_layer_call_and_return_conditional_losses_2617700Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
щ
Еtrace_02╩
-__inference_sampling_10_layer_call_fn_2617706Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0
Д
Жtrace_02х
H__inference_sampling_10_layer_call_and_return_conditional_losses_2617732Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЖtrace_0
╠B╔
%__inference_signature_wrapper_2616913input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
∙BЎ
2__inference_encoder_block_40_layer_call_fn_2617322input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
2__inference_encoder_block_40_layer_call_fn_2617339input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617365input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617391input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
ю
Сtrace_02╧
2__inference_max_pooling2d_40_layer_call_fn_2617737Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0
Й
Тtrace_02ъ
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2617742Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zТtrace_0
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
╕
Уnon_trainable_variables
Фlayers
Хmetrics
 Цlayer_regularization_losses
Чlayer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
ч
Шtrace_0
Щtrace_12м
8__inference_batch_normalization_74_layer_call_fn_2617755
8__inference_batch_normalization_74_layer_call_fn_2617768╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zШtrace_0zЩtrace_1
Э
Ъtrace_0
Ыtrace_12т
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617786
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617804╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0zЫtrace_1
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
∙BЎ
2__inference_encoder_block_41_layer_call_fn_2617408input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
2__inference_encoder_block_41_layer_call_fn_2617425input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617451input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617477input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
д	variables
еtrainable_variables
жregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
ю
жtrace_02╧
2__inference_max_pooling2d_41_layer_call_fn_2617809Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0
Й
зtrace_02ъ
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2617814Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zзtrace_0
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
╕
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
ч
нtrace_0
оtrace_12м
8__inference_batch_normalization_75_layer_call_fn_2617827
8__inference_batch_normalization_75_layer_call_fn_2617840╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zнtrace_0zоtrace_1
Э
пtrace_0
░trace_12т
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617858
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617876╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0z░trace_1
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
∙BЎ
2__inference_encoder_block_42_layer_call_fn_2617494input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
2__inference_encoder_block_42_layer_call_fn_2617511input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617537input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617563input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
║	variables
╗trainable_variables
╝regularization_losses
╛__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╢non_trainable_variables
╖layers
╕metrics
 ╣layer_regularization_losses
║layer_metrics
┴	variables
┬trainable_variables
├regularization_losses
┼__call__
+╞&call_and_return_all_conditional_losses
'╞"call_and_return_conditional_losses"
_generic_user_object
ю
╗trace_02╧
2__inference_max_pooling2d_42_layer_call_fn_2617881Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╗trace_0
Й
╝trace_02ъ
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2617886Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╝trace_0
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
╕
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
╟	variables
╚trainable_variables
╔regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
ч
┬trace_0
├trace_12м
8__inference_batch_normalization_76_layer_call_fn_2617899
8__inference_batch_normalization_76_layer_call_fn_2617912╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┬trace_0z├trace_1
Э
─trace_0
┼trace_12т
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617930
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617948╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0z┼trace_1
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
∙BЎ
2__inference_encoder_block_43_layer_call_fn_2617580input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
∙BЎ
2__inference_encoder_block_43_layer_call_fn_2617597input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617623input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ФBС
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617649input_tensor"п
и▓д
FullArgSpec'
argsЪ
jinput_tensor

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
╫	variables
╪trainable_variables
┘regularization_losses
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
к2зд
Ы▓Ч
FullArgSpec
argsЪ
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
╧layer_metrics
▐	variables
▀trainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
ю
╨trace_02╧
2__inference_max_pooling2d_43_layer_call_fn_2617953Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0
Й
╤trace_02ъ
M__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_2617958Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╤trace_0
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
╕
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
ф	variables
хtrainable_variables
цregularization_losses
ш__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
ч
╫trace_0
╪trace_12м
8__inference_batch_normalization_77_layer_call_fn_2617971
8__inference_batch_normalization_77_layer_call_fn_2617984╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0z╪trace_1
Э
┘trace_0
┌trace_12т
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618002
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618020╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0z┌trace_1
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
╓B╙
,__inference_flatten_10_layer_call_fn_2617654inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ёBю
G__inference_flatten_10_layer_call_and_return_conditional_losses_2617660inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_dense_20_layer_call_fn_2617669inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_dense_20_layer_call_and_return_conditional_losses_2617680inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_dense_21_layer_call_fn_2617689inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
яBь
E__inference_dense_21_layer_call_and_return_conditional_losses_2617700inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
уBр
-__inference_sampling_10_layer_call_fn_2617706inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
H__inference_sampling_10_layer_call_and_return_conditional_losses_2617732inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
2__inference_max_pooling2d_40_layer_call_fn_2617737inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2617742inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
 B№
8__inference_batch_normalization_74_layer_call_fn_2617755inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
8__inference_batch_normalization_74_layer_call_fn_2617768inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617786inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617804inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
2__inference_max_pooling2d_41_layer_call_fn_2617809inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2617814inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
 B№
8__inference_batch_normalization_75_layer_call_fn_2617827inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
8__inference_batch_normalization_75_layer_call_fn_2617840inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617858inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617876inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
2__inference_max_pooling2d_42_layer_call_fn_2617881inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2617886inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
 B№
8__inference_batch_normalization_76_layer_call_fn_2617899inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
8__inference_batch_normalization_76_layer_call_fn_2617912inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617930inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617948inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
▄B┘
2__inference_max_pooling2d_43_layer_call_fn_2617953inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
M__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_2617958inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
 B№
8__inference_batch_normalization_77_layer_call_fn_2617971inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
8__inference_batch_normalization_77_layer_call_fn_2617984inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618002inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЪBЧ
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618020inputs"╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Ш
"__inference__wrapped_model_2615557ё !"#$%&'()*+,:в7
0в-
+К(
input_1         АА
к "ФкР
.
output_1"К
output_1         @
.
output_2"К
output_2         @
.
output_3"К
output_3         @√
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617786гRвO
HвE
;К8
inputs,                           А
p

 
к "GвD
=К:
tensor_0,                           А
Ъ √
S__inference_batch_normalization_74_layer_call_and_return_conditional_losses_2617804гRвO
HвE
;К8
inputs,                           А
p 

 
к "GвD
=К:
tensor_0,                           А
Ъ ╒
8__inference_batch_normalization_74_layer_call_fn_2617755ШRвO
HвE
;К8
inputs,                           А
p

 
к "<К9
unknown,                           А╒
8__inference_batch_normalization_74_layer_call_fn_2617768ШRвO
HвE
;К8
inputs,                           А
p 

 
к "<К9
unknown,                           А√
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617858гRвO
HвE
;К8
inputs,                           А
p

 
к "GвD
=К:
tensor_0,                           А
Ъ √
S__inference_batch_normalization_75_layer_call_and_return_conditional_losses_2617876гRвO
HвE
;К8
inputs,                           А
p 

 
к "GвD
=К:
tensor_0,                           А
Ъ ╒
8__inference_batch_normalization_75_layer_call_fn_2617827ШRвO
HвE
;К8
inputs,                           А
p

 
к "<К9
unknown,                           А╒
8__inference_batch_normalization_75_layer_call_fn_2617840ШRвO
HвE
;К8
inputs,                           А
p 

 
к "<К9
unknown,                           А√
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617930г !"RвO
HвE
;К8
inputs,                           А
p

 
к "GвD
=К:
tensor_0,                           А
Ъ √
S__inference_batch_normalization_76_layer_call_and_return_conditional_losses_2617948г !"RвO
HвE
;К8
inputs,                           А
p 

 
к "GвD
=К:
tensor_0,                           А
Ъ ╒
8__inference_batch_normalization_76_layer_call_fn_2617899Ш !"RвO
HвE
;К8
inputs,                           А
p

 
к "<К9
unknown,                           А╒
8__inference_batch_normalization_76_layer_call_fn_2617912Ш !"RвO
HвE
;К8
inputs,                           А
p 

 
к "<К9
unknown,                           А∙
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618002б%&'(QвN
GвD
:К7
inputs+                           @
p

 
к "FвC
<К9
tensor_0+                           @
Ъ ∙
S__inference_batch_normalization_77_layer_call_and_return_conditional_losses_2618020б%&'(QвN
GвD
:К7
inputs+                           @
p 

 
к "FвC
<К9
tensor_0+                           @
Ъ ╙
8__inference_batch_normalization_77_layer_call_fn_2617971Ц%&'(QвN
GвD
:К7
inputs+                           @
p

 
к ";К8
unknown+                           @╙
8__inference_batch_normalization_77_layer_call_fn_2617984Ц%&'(QвN
GвD
:К7
inputs+                           @
p 

 
к ";К8
unknown+                           @н
E__inference_dense_20_layer_call_and_return_conditional_losses_2617680d)*0в-
&в#
!К
inputs         А 
к ",в)
"К
tensor_0         @
Ъ З
*__inference_dense_20_layer_call_fn_2617669Y)*0в-
&в#
!К
inputs         А 
к "!К
unknown         @н
E__inference_dense_21_layer_call_and_return_conditional_losses_2617700d+,0в-
&в#
!К
inputs         А 
к ",в)
"К
tensor_0         @
Ъ З
*__inference_dense_21_layer_call_fn_2617689Y+,0в-
&в#
!К
inputs         А 
к "!К
unknown         @╓
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617365ДCв@
9в6
0К-
input_tensor         АА
p
к "5в2
+К(
tensor_0         @@А
Ъ ╓
M__inference_encoder_block_40_layer_call_and_return_conditional_losses_2617391ДCв@
9в6
0К-
input_tensor         АА
p 
к "5в2
+К(
tensor_0         @@А
Ъ п
2__inference_encoder_block_40_layer_call_fn_2617322yCв@
9в6
0К-
input_tensor         АА
p
к "*К'
unknown         @@Ап
2__inference_encoder_block_40_layer_call_fn_2617339yCв@
9в6
0К-
input_tensor         АА
p 
к "*К'
unknown         @@А╒
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617451ГBв?
8в5
/К,
input_tensor         @@А
p
к "5в2
+К(
tensor_0           А
Ъ ╒
M__inference_encoder_block_41_layer_call_and_return_conditional_losses_2617477ГBв?
8в5
/К,
input_tensor         @@А
p 
к "5в2
+К(
tensor_0           А
Ъ о
2__inference_encoder_block_41_layer_call_fn_2617408xBв?
8в5
/К,
input_tensor         @@А
p
к "*К'
unknown           Ао
2__inference_encoder_block_41_layer_call_fn_2617425xBв?
8в5
/К,
input_tensor         @@А
p 
к "*К'
unknown           А╒
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617537Г !"Bв?
8в5
/К,
input_tensor           А
p
к "5в2
+К(
tensor_0         А
Ъ ╒
M__inference_encoder_block_42_layer_call_and_return_conditional_losses_2617563Г !"Bв?
8в5
/К,
input_tensor           А
p 
к "5в2
+К(
tensor_0         А
Ъ о
2__inference_encoder_block_42_layer_call_fn_2617494x !"Bв?
8в5
/К,
input_tensor           А
p
к "*К'
unknown         Ао
2__inference_encoder_block_42_layer_call_fn_2617511x !"Bв?
8в5
/К,
input_tensor           А
p 
к "*К'
unknown         А╘
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617623В#$%&'(Bв?
8в5
/К,
input_tensor         А
p
к "4в1
*К'
tensor_0         @
Ъ ╘
M__inference_encoder_block_43_layer_call_and_return_conditional_losses_2617649В#$%&'(Bв?
8в5
/К,
input_tensor         А
p 
к "4в1
*К'
tensor_0         @
Ъ н
2__inference_encoder_block_43_layer_call_fn_2617580w#$%&'(Bв?
8в5
/К,
input_tensor         А
p
к ")К&
unknown         @н
2__inference_encoder_block_43_layer_call_fn_2617597w#$%&'(Bв?
8в5
/К,
input_tensor         А
p 
к ")К&
unknown         @┤
D__inference_encoder_layer_call_and_return_conditional_losses_2616098ы !"#$%&'()*+,JвG
0в-
+К(
input_1         АА
к

trainingp"в|
uЪr
$К!

tensor_0_0         @
$К!

tensor_0_1         @
$К!

tensor_0_2         @
Ъ ┤
D__inference_encoder_layer_call_and_return_conditional_losses_2616272ы !"#$%&'()*+,JвG
0в-
+К(
input_1         АА
к

trainingp "в|
uЪr
$К!

tensor_0_0         @
$К!

tensor_0_1         @
$К!

tensor_0_2         @
Ъ ╣
D__inference_encoder_layer_call_and_return_conditional_losses_2617174Ё !"#$%&'()*+,OвL
5в2
0К-
tensor_input         АА
к

trainingp"в|
uЪr
$К!

tensor_0_0         @
$К!

tensor_0_1         @
$К!

tensor_0_2         @
Ъ ╣
D__inference_encoder_layer_call_and_return_conditional_losses_2617305Ё !"#$%&'()*+,OвL
5в2
0К-
tensor_input         АА
к

trainingp "в|
uЪr
$К!

tensor_0_0         @
$К!

tensor_0_1         @
$К!

tensor_0_2         @
Ъ Й
)__inference_encoder_layer_call_fn_2616408█ !"#$%&'()*+,JвG
0в-
+К(
input_1         АА
к

trainingp"oЪl
"К
tensor_0         @
"К
tensor_1         @
"К
tensor_2         @Й
)__inference_encoder_layer_call_fn_2616543█ !"#$%&'()*+,JвG
0в-
+К(
input_1         АА
к

trainingp "oЪl
"К
tensor_0         @
"К
tensor_1         @
"К
tensor_2         @О
)__inference_encoder_layer_call_fn_2616978р !"#$%&'()*+,OвL
5в2
0К-
tensor_input         АА
к

trainingp"oЪl
"К
tensor_0         @
"К
tensor_1         @
"К
tensor_2         @О
)__inference_encoder_layer_call_fn_2617043р !"#$%&'()*+,OвL
5в2
0К-
tensor_input         АА
к

trainingp "oЪl
"К
tensor_0         @
"К
tensor_1         @
"К
tensor_2         @│
G__inference_flatten_10_layer_call_and_return_conditional_losses_2617660h7в4
-в*
(К%
inputs         @
к "-в*
#К 
tensor_0         А 
Ъ Н
,__inference_flatten_10_layer_call_fn_2617654]7в4
-в*
(К%
inputs         @
к ""К
unknown         А ў
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2617742еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_40_layer_call_fn_2617737ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2617814еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_41_layer_call_fn_2617809ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_42_layer_call_and_return_conditional_losses_2617886еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_42_layer_call_fn_2617881ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ў
M__inference_max_pooling2d_43_layer_call_and_return_conditional_losses_2617958еRвO
HвE
CК@
inputs4                                    
к "OвL
EКB
tensor_04                                    
Ъ ╤
2__inference_max_pooling2d_43_layer_call_fn_2617953ЪRвO
HвE
CК@
inputs4                                    
к "DКA
unknown4                                    ╫
H__inference_sampling_10_layer_call_and_return_conditional_losses_2617732КZвW
PвM
KЪH
"К
inputs_0         @
"К
inputs_1         @
к ",в)
"К
tensor_0         @
Ъ ░
-__inference_sampling_10_layer_call_fn_2617706ZвW
PвM
KЪH
"К
inputs_0         @
"К
inputs_1         @
к "!К
unknown         @ж
%__inference_signature_wrapper_2616913№ !"#$%&'()*+,EвB
в 
;к8
6
input_1+К(
input_1         АА"ФкР
.
output_1"К
output_1         @
.
output_2"К
output_2         @
.
output_3"К
output_3         @