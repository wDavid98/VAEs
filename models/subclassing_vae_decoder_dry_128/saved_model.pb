З─
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
 ѕ"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758в│
ё
decoder/conv2d_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namedecoder/conv2d_66/bias
}
*decoder/conv2d_66/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_66/bias*
_output_shapes
:*
dtype0
ћ
decoder/conv2d_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namedecoder/conv2d_66/kernel
Ї
,decoder/conv2d_66/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_66/kernel*&
_output_shapes
: *
dtype0
ё
decoder/conv2d_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namedecoder/conv2d_65/bias
}
*decoder/conv2d_65/bias/Read/ReadVariableOpReadVariableOpdecoder/conv2d_65/bias*
_output_shapes
: *
dtype0
ћ
decoder/conv2d_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *)
shared_namedecoder/conv2d_65/kernel
Ї
,decoder/conv2d_65/kernel/Read/ReadVariableOpReadVariableOpdecoder/conv2d_65/kernel*&
_output_shapes
:@ *
dtype0
о
?decoder/decoder_block_20/batch_normalization_52/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?decoder/decoder_block_20/batch_normalization_52/moving_variance
¤
Sdecoder/decoder_block_20/batch_normalization_52/moving_variance/Read/ReadVariableOpReadVariableOp?decoder/decoder_block_20/batch_normalization_52/moving_variance*
_output_shapes
:@*
dtype0
╬
;decoder/decoder_block_20/batch_normalization_52/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;decoder/decoder_block_20/batch_normalization_52/moving_mean
К
Odecoder/decoder_block_20/batch_normalization_52/moving_mean/Read/ReadVariableOpReadVariableOp;decoder/decoder_block_20/batch_normalization_52/moving_mean*
_output_shapes
:@*
dtype0
└
4decoder/decoder_block_20/batch_normalization_52/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64decoder/decoder_block_20/batch_normalization_52/beta
╣
Hdecoder/decoder_block_20/batch_normalization_52/beta/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_20/batch_normalization_52/beta*
_output_shapes
:@*
dtype0
┬
5decoder/decoder_block_20/batch_normalization_52/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75decoder/decoder_block_20/batch_normalization_52/gamma
╗
Idecoder/decoder_block_20/batch_normalization_52/gamma/Read/ReadVariableOpReadVariableOp5decoder/decoder_block_20/batch_normalization_52/gamma*
_output_shapes
:@*
dtype0
д
'decoder/decoder_block_20/conv2d_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'decoder/decoder_block_20/conv2d_64/bias
Ъ
;decoder/decoder_block_20/conv2d_64/bias/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_20/conv2d_64/bias*
_output_shapes
:@*
dtype0
и
)decoder/decoder_block_20/conv2d_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ@*:
shared_name+)decoder/decoder_block_20/conv2d_64/kernel
░
=decoder/decoder_block_20/conv2d_64/kernel/Read/ReadVariableOpReadVariableOp)decoder/decoder_block_20/conv2d_64/kernel*'
_output_shapes
:ђ@*
dtype0
О
?decoder/decoder_block_19/batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*P
shared_nameA?decoder/decoder_block_19/batch_normalization_51/moving_variance
л
Sdecoder/decoder_block_19/batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp?decoder/decoder_block_19/batch_normalization_51/moving_variance*
_output_shapes	
:ђ*
dtype0
¤
;decoder/decoder_block_19/batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;decoder/decoder_block_19/batch_normalization_51/moving_mean
╚
Odecoder/decoder_block_19/batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp;decoder/decoder_block_19/batch_normalization_51/moving_mean*
_output_shapes	
:ђ*
dtype0
┴
4decoder/decoder_block_19/batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*E
shared_name64decoder/decoder_block_19/batch_normalization_51/beta
║
Hdecoder/decoder_block_19/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_19/batch_normalization_51/beta*
_output_shapes	
:ђ*
dtype0
├
5decoder/decoder_block_19/batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*F
shared_name75decoder/decoder_block_19/batch_normalization_51/gamma
╝
Idecoder/decoder_block_19/batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOp5decoder/decoder_block_19/batch_normalization_51/gamma*
_output_shapes	
:ђ*
dtype0
Д
'decoder/decoder_block_19/conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'decoder/decoder_block_19/conv2d_63/bias
а
;decoder/decoder_block_19/conv2d_63/bias/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_19/conv2d_63/bias*
_output_shapes	
:ђ*
dtype0
И
)decoder/decoder_block_19/conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*:
shared_name+)decoder/decoder_block_19/conv2d_63/kernel
▒
=decoder/decoder_block_19/conv2d_63/kernel/Read/ReadVariableOpReadVariableOp)decoder/decoder_block_19/conv2d_63/kernel*(
_output_shapes
:ђђ*
dtype0
О
?decoder/decoder_block_18/batch_normalization_50/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*P
shared_nameA?decoder/decoder_block_18/batch_normalization_50/moving_variance
л
Sdecoder/decoder_block_18/batch_normalization_50/moving_variance/Read/ReadVariableOpReadVariableOp?decoder/decoder_block_18/batch_normalization_50/moving_variance*
_output_shapes	
:ђ*
dtype0
¤
;decoder/decoder_block_18/batch_normalization_50/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;decoder/decoder_block_18/batch_normalization_50/moving_mean
╚
Odecoder/decoder_block_18/batch_normalization_50/moving_mean/Read/ReadVariableOpReadVariableOp;decoder/decoder_block_18/batch_normalization_50/moving_mean*
_output_shapes	
:ђ*
dtype0
┴
4decoder/decoder_block_18/batch_normalization_50/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*E
shared_name64decoder/decoder_block_18/batch_normalization_50/beta
║
Hdecoder/decoder_block_18/batch_normalization_50/beta/Read/ReadVariableOpReadVariableOp4decoder/decoder_block_18/batch_normalization_50/beta*
_output_shapes	
:ђ*
dtype0
├
5decoder/decoder_block_18/batch_normalization_50/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*F
shared_name75decoder/decoder_block_18/batch_normalization_50/gamma
╝
Idecoder/decoder_block_18/batch_normalization_50/gamma/Read/ReadVariableOpReadVariableOp5decoder/decoder_block_18/batch_normalization_50/gamma*
_output_shapes	
:ђ*
dtype0
Д
'decoder/decoder_block_18/conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'decoder/decoder_block_18/conv2d_62/bias
а
;decoder/decoder_block_18/conv2d_62/bias/Read/ReadVariableOpReadVariableOp'decoder/decoder_block_18/conv2d_62/bias*
_output_shapes	
:ђ*
dtype0
И
)decoder/decoder_block_18/conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*:
shared_name+)decoder/decoder_block_18/conv2d_62/kernel
▒
=decoder/decoder_block_18/conv2d_62/kernel/Read/ReadVariableOpReadVariableOp)decoder/decoder_block_18/conv2d_62/kernel*(
_output_shapes
:ђђ*
dtype0
ё
decoder/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*&
shared_namedecoder/dense_22/bias
}
)decoder/dense_22/bias/Read/ReadVariableOpReadVariableOpdecoder/dense_22/bias*
_output_shapes

:ђђ*
dtype0
Ї
decoder/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђђ*(
shared_namedecoder/dense_22/kernel
є
+decoder/dense_22/kernel/Read/ReadVariableOpReadVariableOpdecoder/dense_22/kernel*!
_output_shapes
:ђђђ*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:         ђ*
dtype0*
shape:         ђ
├
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1decoder/dense_22/kerneldecoder/dense_22/bias)decoder/decoder_block_18/conv2d_62/kernel'decoder/decoder_block_18/conv2d_62/bias5decoder/decoder_block_18/batch_normalization_50/gamma4decoder/decoder_block_18/batch_normalization_50/beta;decoder/decoder_block_18/batch_normalization_50/moving_mean?decoder/decoder_block_18/batch_normalization_50/moving_variance)decoder/decoder_block_19/conv2d_63/kernel'decoder/decoder_block_19/conv2d_63/bias5decoder/decoder_block_19/batch_normalization_51/gamma4decoder/decoder_block_19/batch_normalization_51/beta;decoder/decoder_block_19/batch_normalization_51/moving_mean?decoder/decoder_block_19/batch_normalization_51/moving_variance)decoder/decoder_block_20/conv2d_64/kernel'decoder/decoder_block_20/conv2d_64/bias5decoder/decoder_block_20/batch_normalization_52/gamma4decoder/decoder_block_20/batch_normalization_52/beta;decoder/decoder_block_20/batch_normalization_52/moving_mean?decoder/decoder_block_20/batch_normalization_52/moving_variancedecoder/conv2d_65/kerneldecoder/conv2d_65/biasdecoder/conv2d_66/kerneldecoder/conv2d_66/bias*$
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
%__inference_signature_wrapper_1937764

NoOpNoOp
­e
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Фe
valueАeBъe BЌe
Ф
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
║
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
і
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
░
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
д
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

kernel
bias*
ј
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
б
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jbn*
ј
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
б
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Wconv
Xbn*
ј
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
б
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
econv
fbn*
╚
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

&kernel
'bias
 m_jit_compiled_convolution_op*
╚
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
WQ
VARIABLE_VALUEdecoder/dense_22/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdecoder/dense_22/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)decoder/decoder_block_18/conv2d_62/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'decoder/decoder_block_18/conv2d_62/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5decoder/decoder_block_18/batch_normalization_50/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4decoder/decoder_block_18/batch_normalization_50/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;decoder/decoder_block_18/batch_normalization_50/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?decoder/decoder_block_18/batch_normalization_50/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)decoder/decoder_block_19/conv2d_63/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'decoder/decoder_block_19/conv2d_63/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5decoder/decoder_block_19/batch_normalization_51/gamma'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4decoder/decoder_block_19/batch_normalization_51/beta'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;decoder/decoder_block_19/batch_normalization_51/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE?decoder/decoder_block_19/batch_normalization_51/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)decoder/decoder_block_20/conv2d_64/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'decoder/decoder_block_20/conv2d_64/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5decoder/decoder_block_20/batch_normalization_52/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4decoder/decoder_block_20/batch_normalization_52/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;decoder/decoder_block_20/batch_normalization_52/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE?decoder/decoder_block_20/batch_normalization_52/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_65/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_65/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdecoder/conv2d_66/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdecoder/conv2d_66/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
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
Њ
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
Њ
}non_trainable_variables

~layers
metrics
 ђlayer_regularization_losses
Ђlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

ѓtrace_0* 

Ѓtrace_0* 
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
ў
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

Ѕtrace_0
іtrace_1* 

Іtrace_0
їtrace_1* 
¤
Ї	variables
јtrainable_variables
Јregularization_losses
љ	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses

kernel
bias
!Њ_jit_compiled_convolution_op*
▄
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
	џaxis
	gamma
beta
moving_mean
moving_variance*
* 
* 
* 
ќ
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

аtrace_0* 

Аtrace_0* 
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
ў
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

Дtrace_0
еtrace_1* 

Еtrace_0
фtrace_1* 
¤
Ф	variables
гtrainable_variables
Гregularization_losses
«	keras_api
»__call__
+░&call_and_return_all_conditional_losses

kernel
bias
!▒_jit_compiled_convolution_op*
▄
▓	variables
│trainable_variables
┤regularization_losses
х	keras_api
Х__call__
+и&call_and_return_all_conditional_losses
	Иaxis
	gamma
beta
moving_mean
moving_variance*
* 
* 
* 
ќ
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
йlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

Йtrace_0* 

┐trace_0* 
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
ў
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

┼trace_0
кtrace_1* 

Кtrace_0
╚trace_1* 
¤
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses

 kernel
!bias
!¤_jit_compiled_convolution_op*
▄
л	variables
Лtrainable_variables
мregularization_losses
М	keras_api
н__call__
+Н&call_and_return_all_conditional_losses
	оaxis
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
ў
Оnon_trainable_variables
пlayers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

▄trace_0* 

Пtrace_0* 
* 

(0
)1*

(0
)1*
* 
ў
яnon_trainable_variables
▀layers
Яmetrics
 рlayer_regularization_losses
Рlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

сtrace_0* 

Сtrace_0* 
* 
* 
* 
* 
* 
* 
* 
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
ъ
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
Ї	variables
јtrainable_variables
Јregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses*
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
ъ
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*

№trace_0
­trace_1* 

ыtrace_0
Ыtrace_1* 
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
ъ
зnon_trainable_variables
Зlayers
шmetrics
 Шlayer_regularization_losses
эlayer_metrics
Ф	variables
гtrainable_variables
Гregularization_losses
»__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses*
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
ъ
Эnon_trainable_variables
щlayers
Щmetrics
 чlayer_regularization_losses
Чlayer_metrics
▓	variables
│trainable_variables
┤regularization_losses
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses*

§trace_0
■trace_1* 

 trace_0
ђtrace_1* 
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
ъ
Ђnon_trainable_variables
ѓlayers
Ѓmetrics
 ёlayer_regularization_losses
Ёlayer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses*
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
ъ
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
л	variables
Лtrainable_variables
мregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses*

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
і
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedecoder/dense_22/kerneldecoder/dense_22/bias)decoder/decoder_block_18/conv2d_62/kernel'decoder/decoder_block_18/conv2d_62/bias5decoder/decoder_block_18/batch_normalization_50/gamma4decoder/decoder_block_18/batch_normalization_50/beta;decoder/decoder_block_18/batch_normalization_50/moving_mean?decoder/decoder_block_18/batch_normalization_50/moving_variance)decoder/decoder_block_19/conv2d_63/kernel'decoder/decoder_block_19/conv2d_63/bias5decoder/decoder_block_19/batch_normalization_51/gamma4decoder/decoder_block_19/batch_normalization_51/beta;decoder/decoder_block_19/batch_normalization_51/moving_mean?decoder/decoder_block_19/batch_normalization_51/moving_variance)decoder/decoder_block_20/conv2d_64/kernel'decoder/decoder_block_20/conv2d_64/bias5decoder/decoder_block_20/batch_normalization_52/gamma4decoder/decoder_block_20/batch_normalization_52/beta;decoder/decoder_block_20/batch_normalization_52/moving_mean?decoder/decoder_block_20/batch_normalization_52/moving_variancedecoder/conv2d_65/kerneldecoder/conv2d_65/biasdecoder/conv2d_66/kerneldecoder/conv2d_66/biasConst*%
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
 __inference__traced_save_1938790
Ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedecoder/dense_22/kerneldecoder/dense_22/bias)decoder/decoder_block_18/conv2d_62/kernel'decoder/decoder_block_18/conv2d_62/bias5decoder/decoder_block_18/batch_normalization_50/gamma4decoder/decoder_block_18/batch_normalization_50/beta;decoder/decoder_block_18/batch_normalization_50/moving_mean?decoder/decoder_block_18/batch_normalization_50/moving_variance)decoder/decoder_block_19/conv2d_63/kernel'decoder/decoder_block_19/conv2d_63/bias5decoder/decoder_block_19/batch_normalization_51/gamma4decoder/decoder_block_19/batch_normalization_51/beta;decoder/decoder_block_19/batch_normalization_51/moving_mean?decoder/decoder_block_19/batch_normalization_51/moving_variance)decoder/decoder_block_20/conv2d_64/kernel'decoder/decoder_block_20/conv2d_64/bias5decoder/decoder_block_20/batch_normalization_52/gamma4decoder/decoder_block_20/batch_normalization_52/beta;decoder/decoder_block_20/batch_normalization_52/moving_mean?decoder/decoder_block_20/batch_normalization_52/moving_variancedecoder/conv2d_65/kerneldecoder/conv2d_65/biasdecoder/conv2d_66/kerneldecoder/conv2d_66/bias*$
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
#__inference__traced_restore_1938872цу
Ё

ў
2__inference_decoder_block_19_layer_call_fn_1938229
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937008і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
ћ
ќ
%__inference_signature_wrapper_1937764
input_1
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ%

unknown_13:ђ@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: $

unknown_21: 

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
"__inference__wrapped_model_1936670y
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
┴
N
2__inference_up_sampling2d_18_layer_call_fn_1938099

inputs
identityЯ
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1936683Ѓ
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
ьБ
Т
"__inference__wrapped_model_1936670
input_1D
/decoder_dense_22_matmul_readvariableop_resource:ђђђ@
0decoder_dense_22_biasadd_readvariableop_resource:
ђђ]
Adecoder_decoder_block_18_conv2d_62_conv2d_readvariableop_resource:ђђQ
Bdecoder_decoder_block_18_conv2d_62_biasadd_readvariableop_resource:	ђV
Gdecoder_decoder_block_18_batch_normalization_50_readvariableop_resource:	ђX
Idecoder_decoder_block_18_batch_normalization_50_readvariableop_1_resource:	ђg
Xdecoder_decoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђi
Zdecoder_decoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Adecoder_decoder_block_19_conv2d_63_conv2d_readvariableop_resource:ђђQ
Bdecoder_decoder_block_19_conv2d_63_biasadd_readvariableop_resource:	ђV
Gdecoder_decoder_block_19_batch_normalization_51_readvariableop_resource:	ђX
Idecoder_decoder_block_19_batch_normalization_51_readvariableop_1_resource:	ђg
Xdecoder_decoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђi
Zdecoder_decoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђ\
Adecoder_decoder_block_20_conv2d_64_conv2d_readvariableop_resource:ђ@P
Bdecoder_decoder_block_20_conv2d_64_biasadd_readvariableop_resource:@U
Gdecoder_decoder_block_20_batch_normalization_52_readvariableop_resource:@W
Idecoder_decoder_block_20_batch_normalization_52_readvariableop_1_resource:@f
Xdecoder_decoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@h
Zdecoder_decoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@J
0decoder_conv2d_65_conv2d_readvariableop_resource:@ ?
1decoder_conv2d_65_biasadd_readvariableop_resource: J
0decoder_conv2d_66_conv2d_readvariableop_resource: ?
1decoder_conv2d_66_biasadd_readvariableop_resource:
identityѕб(decoder/conv2d_65/BiasAdd/ReadVariableOpб'decoder/conv2d_65/Conv2D/ReadVariableOpб(decoder/conv2d_66/BiasAdd/ReadVariableOpб'decoder/conv2d_66/Conv2D/ReadVariableOpбOdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpбQdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б>decoder/decoder_block_18/batch_normalization_50/ReadVariableOpб@decoder/decoder_block_18/batch_normalization_50/ReadVariableOp_1б9decoder/decoder_block_18/conv2d_62/BiasAdd/ReadVariableOpб8decoder/decoder_block_18/conv2d_62/Conv2D/ReadVariableOpбOdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpбQdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б>decoder/decoder_block_19/batch_normalization_51/ReadVariableOpб@decoder/decoder_block_19/batch_normalization_51/ReadVariableOp_1б9decoder/decoder_block_19/conv2d_63/BiasAdd/ReadVariableOpб8decoder/decoder_block_19/conv2d_63/Conv2D/ReadVariableOpбOdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpбQdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б>decoder/decoder_block_20/batch_normalization_52/ReadVariableOpб@decoder/decoder_block_20/batch_normalization_52/ReadVariableOp_1б9decoder/decoder_block_20/conv2d_64/BiasAdd/ReadVariableOpб8decoder/decoder_block_20/conv2d_64/Conv2D/ReadVariableOpб'decoder/dense_22/BiasAdd/ReadVariableOpб&decoder/dense_22/MatMul/ReadVariableOpЎ
&decoder/dense_22/MatMul/ReadVariableOpReadVariableOp/decoder_dense_22_matmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0ј
decoder/dense_22/MatMulMatMulinput_1.decoder/dense_22/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђќ
'decoder/dense_22/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_22_biasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0Ф
decoder/dense_22/BiasAddBiasAdd!decoder/dense_22/MatMul:product:0/decoder/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђt
decoder/dense_22/ReluRelu!decoder/dense_22/BiasAdd:output:0*
T0*)
_output_shapes
:         ђђn
decoder/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             џ
decoder/ReshapeReshape#decoder/dense_22/Relu:activations:0decoder/Reshape/shape:output:0*
T0*0
_output_shapes
:         ђo
decoder/up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      q
 decoder/up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ю
decoder/up_sampling2d_18/mulMul'decoder/up_sampling2d_18/Const:output:0)decoder/up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:▀
5decoder/up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighbordecoder/Reshape:output:0 decoder/up_sampling2d_18/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(─
8decoder/decoder_block_18/conv2d_62/Conv2D/ReadVariableOpReadVariableOpAdecoder_decoder_block_18_conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0а
)decoder/decoder_block_18/conv2d_62/Conv2DConv2DFdecoder/up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:0@decoder/decoder_block_18/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
╣
9decoder/decoder_block_18/conv2d_62/BiasAdd/ReadVariableOpReadVariableOpBdecoder_decoder_block_18_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0у
*decoder/decoder_block_18/conv2d_62/BiasAddBiasAdd2decoder/decoder_block_18/conv2d_62/Conv2D:output:0Adecoder/decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЪ
'decoder/decoder_block_18/conv2d_62/ReluRelu3decoder/decoder_block_18/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ├
>decoder/decoder_block_18/batch_normalization_50/ReadVariableOpReadVariableOpGdecoder_decoder_block_18_batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0К
@decoder/decoder_block_18/batch_normalization_50/ReadVariableOp_1ReadVariableOpIdecoder_decoder_block_18_batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0т
Odecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOpXdecoder_decoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
Qdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZdecoder_decoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0┌
@decoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3FusedBatchNormV35decoder/decoder_block_18/conv2d_62/Relu:activations:0Fdecoder/decoder_block_18/batch_normalization_50/ReadVariableOp:value:0Hdecoder/decoder_block_18/batch_normalization_50/ReadVariableOp_1:value:0Wdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0Ydecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( o
decoder/up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      q
 decoder/up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ю
decoder/up_sampling2d_19/mulMul'decoder/up_sampling2d_19/Const:output:0)decoder/up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:І
5decoder/up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighborDdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3:y:0 decoder/up_sampling2d_19/mul:z:0*
T0*0
_output_shapes
:           ђ*
half_pixel_centers(─
8decoder/decoder_block_19/conv2d_63/Conv2D/ReadVariableOpReadVariableOpAdecoder_decoder_block_19_conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0а
)decoder/decoder_block_19/conv2d_63/Conv2DConv2DFdecoder/up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:0@decoder/decoder_block_19/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
╣
9decoder/decoder_block_19/conv2d_63/BiasAdd/ReadVariableOpReadVariableOpBdecoder_decoder_block_19_conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0у
*decoder/decoder_block_19/conv2d_63/BiasAddBiasAdd2decoder/decoder_block_19/conv2d_63/Conv2D:output:0Adecoder/decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђЪ
'decoder/decoder_block_19/conv2d_63/ReluRelu3decoder/decoder_block_19/conv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ├
>decoder/decoder_block_19/batch_normalization_51/ReadVariableOpReadVariableOpGdecoder_decoder_block_19_batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0К
@decoder/decoder_block_19/batch_normalization_51/ReadVariableOp_1ReadVariableOpIdecoder_decoder_block_19_batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0т
Odecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpXdecoder_decoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
Qdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZdecoder_decoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0┌
@decoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3FusedBatchNormV35decoder/decoder_block_19/conv2d_63/Relu:activations:0Fdecoder/decoder_block_19/batch_normalization_51/ReadVariableOp:value:0Hdecoder/decoder_block_19/batch_normalization_51/ReadVariableOp_1:value:0Wdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0Ydecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( o
decoder/up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        q
 decoder/up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ю
decoder/up_sampling2d_20/mulMul'decoder/up_sampling2d_20/Const:output:0)decoder/up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:Ї
5decoder/up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighborDdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3:y:0 decoder/up_sampling2d_20/mul:z:0*
T0*2
_output_shapes 
:         ђђђ*
half_pixel_centers(├
8decoder/decoder_block_20/conv2d_64/Conv2D/ReadVariableOpReadVariableOpAdecoder_decoder_block_20_conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0А
)decoder/decoder_block_20/conv2d_64/Conv2DConv2DFdecoder/up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:0@decoder/decoder_block_20/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@*
paddingSAME*
strides
И
9decoder/decoder_block_20/conv2d_64/BiasAdd/ReadVariableOpReadVariableOpBdecoder_decoder_block_20_conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0У
*decoder/decoder_block_20/conv2d_64/BiasAddBiasAdd2decoder/decoder_block_20/conv2d_64/Conv2D:output:0Adecoder/decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@а
'decoder/decoder_block_20/conv2d_64/ReluRelu3decoder/decoder_block_20/conv2d_64/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@┬
>decoder/decoder_block_20/batch_normalization_52/ReadVariableOpReadVariableOpGdecoder_decoder_block_20_batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0к
@decoder/decoder_block_20/batch_normalization_52/ReadVariableOp_1ReadVariableOpIdecoder_decoder_block_20_batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0С
Odecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOpXdecoder_decoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0У
Qdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZdecoder_decoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0О
@decoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3FusedBatchNormV35decoder/decoder_block_20/conv2d_64/Relu:activations:0Fdecoder/decoder_block_20/batch_normalization_52/ReadVariableOp:value:0Hdecoder/decoder_block_20/batch_normalization_52/ReadVariableOp_1:value:0Wdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0Ydecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( а
'decoder/conv2d_65/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_65_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0§
decoder/conv2d_65/Conv2DConv2DDdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3:y:0/decoder/conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ *
paddingSAME*
strides
ќ
(decoder/conv2d_65/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_65_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0х
decoder/conv2d_65/BiasAddBiasAdd!decoder/conv2d_65/Conv2D:output:00decoder/conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ ~
decoder/conv2d_65/ReluRelu"decoder/conv2d_65/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ а
'decoder/conv2d_66/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0П
decoder/conv2d_66/Conv2DConv2D$decoder/conv2d_65/Relu:activations:0/decoder/conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
ќ
(decoder/conv2d_66/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2d_66_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
decoder/conv2d_66/BiasAddBiasAdd!decoder/conv2d_66/Conv2D:output:00decoder/conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђё
decoder/conv2d_66/SigmoidSigmoid"decoder/conv2d_66/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђv
IdentityIdentitydecoder/conv2d_66/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђд
NoOpNoOp)^decoder/conv2d_65/BiasAdd/ReadVariableOp(^decoder/conv2d_65/Conv2D/ReadVariableOp)^decoder/conv2d_66/BiasAdd/ReadVariableOp(^decoder/conv2d_66/Conv2D/ReadVariableOpP^decoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpR^decoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1?^decoder/decoder_block_18/batch_normalization_50/ReadVariableOpA^decoder/decoder_block_18/batch_normalization_50/ReadVariableOp_1:^decoder/decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp9^decoder/decoder_block_18/conv2d_62/Conv2D/ReadVariableOpP^decoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpR^decoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?^decoder/decoder_block_19/batch_normalization_51/ReadVariableOpA^decoder/decoder_block_19/batch_normalization_51/ReadVariableOp_1:^decoder/decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp9^decoder/decoder_block_19/conv2d_63/Conv2D/ReadVariableOpP^decoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpR^decoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1?^decoder/decoder_block_20/batch_normalization_52/ReadVariableOpA^decoder/decoder_block_20/batch_normalization_52/ReadVariableOp_1:^decoder/decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp9^decoder/decoder_block_20/conv2d_64/Conv2D/ReadVariableOp(^decoder/dense_22/BiasAdd/ReadVariableOp'^decoder/dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2T
(decoder/conv2d_65/BiasAdd/ReadVariableOp(decoder/conv2d_65/BiasAdd/ReadVariableOp2R
'decoder/conv2d_65/Conv2D/ReadVariableOp'decoder/conv2d_65/Conv2D/ReadVariableOp2T
(decoder/conv2d_66/BiasAdd/ReadVariableOp(decoder/conv2d_66/BiasAdd/ReadVariableOp2R
'decoder/conv2d_66/Conv2D/ReadVariableOp'decoder/conv2d_66/Conv2D/ReadVariableOp2д
Qdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Qdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12б
Odecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpOdecoder/decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp2ё
@decoder/decoder_block_18/batch_normalization_50/ReadVariableOp_1@decoder/decoder_block_18/batch_normalization_50/ReadVariableOp_12ђ
>decoder/decoder_block_18/batch_normalization_50/ReadVariableOp>decoder/decoder_block_18/batch_normalization_50/ReadVariableOp2v
9decoder/decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp9decoder/decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp2t
8decoder/decoder_block_18/conv2d_62/Conv2D/ReadVariableOp8decoder/decoder_block_18/conv2d_62/Conv2D/ReadVariableOp2д
Qdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1Qdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12б
Odecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpOdecoder/decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2ё
@decoder/decoder_block_19/batch_normalization_51/ReadVariableOp_1@decoder/decoder_block_19/batch_normalization_51/ReadVariableOp_12ђ
>decoder/decoder_block_19/batch_normalization_51/ReadVariableOp>decoder/decoder_block_19/batch_normalization_51/ReadVariableOp2v
9decoder/decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp9decoder/decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp2t
8decoder/decoder_block_19/conv2d_63/Conv2D/ReadVariableOp8decoder/decoder_block_19/conv2d_63/Conv2D/ReadVariableOp2д
Qdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1Qdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12б
Odecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpOdecoder/decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp2ё
@decoder/decoder_block_20/batch_normalization_52/ReadVariableOp_1@decoder/decoder_block_20/batch_normalization_52/ReadVariableOp_12ђ
>decoder/decoder_block_20/batch_normalization_52/ReadVariableOp>decoder/decoder_block_20/batch_normalization_52/ReadVariableOp2v
9decoder/decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp9decoder/decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp2t
8decoder/decoder_block_20/conv2d_64/Conv2D/ReadVariableOp8decoder/decoder_block_20/conv2d_64/Conv2D/ReadVariableOp2R
'decoder/dense_22/BiasAdd/ReadVariableOp'decoder/dense_22/BiasAdd/ReadVariableOp2P
&decoder/dense_22/MatMul/ReadVariableOp&decoder/dense_22/MatMul/ReadVariableOp:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
ѕ
┬
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1936874

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
┤┴
Д
 __inference__traced_save_1938790
file_prefixC
.read_disablecopyonread_decoder_dense_22_kernel:ђђђ>
.read_1_disablecopyonread_decoder_dense_22_bias:
ђђ^
Bread_2_disablecopyonread_decoder_decoder_block_18_conv2d_62_kernel:ђђO
@read_3_disablecopyonread_decoder_decoder_block_18_conv2d_62_bias:	ђ]
Nread_4_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_gamma:	ђ\
Mread_5_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_beta:	ђc
Tread_6_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_moving_mean:	ђg
Xread_7_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_moving_variance:	ђ^
Bread_8_disablecopyonread_decoder_decoder_block_19_conv2d_63_kernel:ђђO
@read_9_disablecopyonread_decoder_decoder_block_19_conv2d_63_bias:	ђ^
Oread_10_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_gamma:	ђ]
Nread_11_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_beta:	ђd
Uread_12_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_moving_mean:	ђh
Yread_13_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_moving_variance:	ђ^
Cread_14_disablecopyonread_decoder_decoder_block_20_conv2d_64_kernel:ђ@O
Aread_15_disablecopyonread_decoder_decoder_block_20_conv2d_64_bias:@]
Oread_16_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_gamma:@\
Nread_17_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_beta:@c
Uread_18_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_moving_mean:@g
Yread_19_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_moving_variance:@L
2read_20_disablecopyonread_decoder_conv2d_65_kernel:@ >
0read_21_disablecopyonread_decoder_conv2d_65_bias: L
2read_22_disablecopyonread_decoder_conv2d_66_kernel: >
0read_23_disablecopyonread_decoder_conv2d_66_bias:
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
: ђ
Read/DisableCopyOnReadDisableCopyOnRead.read_disablecopyonread_decoder_dense_22_kernel"/device:CPU:0*
_output_shapes
 Г
Read/ReadVariableOpReadVariableOp.read_disablecopyonread_decoder_dense_22_kernel^Read/DisableCopyOnRead"/device:CPU:0*!
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
:ђђђѓ
Read_1/DisableCopyOnReadDisableCopyOnRead.read_1_disablecopyonread_decoder_dense_22_bias"/device:CPU:0*
_output_shapes
 г
Read_1/ReadVariableOpReadVariableOp.read_1_disablecopyonread_decoder_dense_22_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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

:ђђќ
Read_2/DisableCopyOnReadDisableCopyOnReadBread_2_disablecopyonread_decoder_decoder_block_18_conv2d_62_kernel"/device:CPU:0*
_output_shapes
 ╠
Read_2/ReadVariableOpReadVariableOpBread_2_disablecopyonread_decoder_decoder_block_18_conv2d_62_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:ђђ*
dtype0w

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ђђm

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*(
_output_shapes
:ђђћ
Read_3/DisableCopyOnReadDisableCopyOnRead@read_3_disablecopyonread_decoder_decoder_block_18_conv2d_62_bias"/device:CPU:0*
_output_shapes
 й
Read_3/ReadVariableOpReadVariableOp@read_3_disablecopyonread_decoder_decoder_block_18_conv2d_62_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђб
Read_4/DisableCopyOnReadDisableCopyOnReadNread_4_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_4/ReadVariableOpReadVariableOpNread_4_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_gamma^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђА
Read_5/DisableCopyOnReadDisableCopyOnReadMread_5_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_beta"/device:CPU:0*
_output_shapes
 ╩
Read_5/ReadVariableOpReadVariableOpMread_5_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_beta^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђе
Read_6/DisableCopyOnReadDisableCopyOnReadTread_6_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_moving_mean"/device:CPU:0*
_output_shapes
 Л
Read_6/ReadVariableOpReadVariableOpTread_6_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_moving_mean^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђг
Read_7/DisableCopyOnReadDisableCopyOnReadXread_7_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_7/ReadVariableOpReadVariableOpXread_7_disablecopyonread_decoder_decoder_block_18_batch_normalization_50_moving_variance^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђќ
Read_8/DisableCopyOnReadDisableCopyOnReadBread_8_disablecopyonread_decoder_decoder_block_19_conv2d_63_kernel"/device:CPU:0*
_output_shapes
 ╠
Read_8/ReadVariableOpReadVariableOpBread_8_disablecopyonread_decoder_decoder_block_19_conv2d_63_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:ђђ*
dtype0x
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ђђo
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*(
_output_shapes
:ђђћ
Read_9/DisableCopyOnReadDisableCopyOnRead@read_9_disablecopyonread_decoder_decoder_block_19_conv2d_63_bias"/device:CPU:0*
_output_shapes
 й
Read_9/ReadVariableOpReadVariableOp@read_9_disablecopyonread_decoder_decoder_block_19_conv2d_63_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђц
Read_10/DisableCopyOnReadDisableCopyOnReadOread_10_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_gamma"/device:CPU:0*
_output_shapes
 ╬
Read_10/ReadVariableOpReadVariableOpOread_10_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_gamma^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђБ
Read_11/DisableCopyOnReadDisableCopyOnReadNread_11_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_beta"/device:CPU:0*
_output_shapes
 ═
Read_11/ReadVariableOpReadVariableOpNread_11_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_beta^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђф
Read_12/DisableCopyOnReadDisableCopyOnReadUread_12_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_moving_mean"/device:CPU:0*
_output_shapes
 н
Read_12/ReadVariableOpReadVariableOpUread_12_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_moving_mean^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђ«
Read_13/DisableCopyOnReadDisableCopyOnReadYread_13_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_moving_variance"/device:CPU:0*
_output_shapes
 п
Read_13/ReadVariableOpReadVariableOpYread_13_disablecopyonread_decoder_decoder_block_19_batch_normalization_51_moving_variance^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђў
Read_14/DisableCopyOnReadDisableCopyOnReadCread_14_disablecopyonread_decoder_decoder_block_20_conv2d_64_kernel"/device:CPU:0*
_output_shapes
 ╬
Read_14/ReadVariableOpReadVariableOpCread_14_disablecopyonread_decoder_decoder_block_20_conv2d_64_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:ђ@*
dtype0x
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:ђ@n
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*'
_output_shapes
:ђ@ќ
Read_15/DisableCopyOnReadDisableCopyOnReadAread_15_disablecopyonread_decoder_decoder_block_20_conv2d_64_bias"/device:CPU:0*
_output_shapes
 ┐
Read_15/ReadVariableOpReadVariableOpAread_15_disablecopyonread_decoder_decoder_block_20_conv2d_64_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@ц
Read_16/DisableCopyOnReadDisableCopyOnReadOread_16_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_gamma"/device:CPU:0*
_output_shapes
 ═
Read_16/ReadVariableOpReadVariableOpOread_16_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_gamma^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@Б
Read_17/DisableCopyOnReadDisableCopyOnReadNread_17_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_beta"/device:CPU:0*
_output_shapes
 ╠
Read_17/ReadVariableOpReadVariableOpNread_17_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_beta^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@ф
Read_18/DisableCopyOnReadDisableCopyOnReadUread_18_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_moving_mean"/device:CPU:0*
_output_shapes
 М
Read_18/ReadVariableOpReadVariableOpUread_18_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_moving_mean^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:@«
Read_19/DisableCopyOnReadDisableCopyOnReadYread_19_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_moving_variance"/device:CPU:0*
_output_shapes
 О
Read_19/ReadVariableOpReadVariableOpYread_19_disablecopyonread_decoder_decoder_block_20_batch_normalization_52_moving_variance^Read_19/DisableCopyOnRead"/device:CPU:0*
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
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_decoder_conv2d_65_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_decoder_conv2d_65_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@ *
dtype0w
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@ m
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*&
_output_shapes
:@ Ё
Read_21/DisableCopyOnReadDisableCopyOnRead0read_21_disablecopyonread_decoder_conv2d_65_bias"/device:CPU:0*
_output_shapes
 «
Read_21/ReadVariableOpReadVariableOp0read_21_disablecopyonread_decoder_conv2d_65_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: Є
Read_22/DisableCopyOnReadDisableCopyOnRead2read_22_disablecopyonread_decoder_conv2d_66_kernel"/device:CPU:0*
_output_shapes
 ╝
Read_22/ReadVariableOpReadVariableOp2read_22_disablecopyonread_decoder_conv2d_66_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*&
_output_shapes
: Ё
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_decoder_conv2d_66_bias"/device:CPU:0*
_output_shapes
 «
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_decoder_conv2d_66_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
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
пo
Ќ
#__inference__traced_restore_1938872
file_prefix=
(assignvariableop_decoder_dense_22_kernel:ђђђ8
(assignvariableop_1_decoder_dense_22_bias:
ђђX
<assignvariableop_2_decoder_decoder_block_18_conv2d_62_kernel:ђђI
:assignvariableop_3_decoder_decoder_block_18_conv2d_62_bias:	ђW
Hassignvariableop_4_decoder_decoder_block_18_batch_normalization_50_gamma:	ђV
Gassignvariableop_5_decoder_decoder_block_18_batch_normalization_50_beta:	ђ]
Nassignvariableop_6_decoder_decoder_block_18_batch_normalization_50_moving_mean:	ђa
Rassignvariableop_7_decoder_decoder_block_18_batch_normalization_50_moving_variance:	ђX
<assignvariableop_8_decoder_decoder_block_19_conv2d_63_kernel:ђђI
:assignvariableop_9_decoder_decoder_block_19_conv2d_63_bias:	ђX
Iassignvariableop_10_decoder_decoder_block_19_batch_normalization_51_gamma:	ђW
Hassignvariableop_11_decoder_decoder_block_19_batch_normalization_51_beta:	ђ^
Oassignvariableop_12_decoder_decoder_block_19_batch_normalization_51_moving_mean:	ђb
Sassignvariableop_13_decoder_decoder_block_19_batch_normalization_51_moving_variance:	ђX
=assignvariableop_14_decoder_decoder_block_20_conv2d_64_kernel:ђ@I
;assignvariableop_15_decoder_decoder_block_20_conv2d_64_bias:@W
Iassignvariableop_16_decoder_decoder_block_20_batch_normalization_52_gamma:@V
Hassignvariableop_17_decoder_decoder_block_20_batch_normalization_52_beta:@]
Oassignvariableop_18_decoder_decoder_block_20_batch_normalization_52_moving_mean:@a
Sassignvariableop_19_decoder_decoder_block_20_batch_normalization_52_moving_variance:@F
,assignvariableop_20_decoder_conv2d_65_kernel:@ 8
*assignvariableop_21_decoder_conv2d_65_bias: F
,assignvariableop_22_decoder_conv2d_66_kernel: 8
*assignvariableop_23_decoder_conv2d_66_bias:
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
:╗
AssignVariableOpAssignVariableOp(assignvariableop_decoder_dense_22_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_1AssignVariableOp(assignvariableop_1_decoder_dense_22_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_2AssignVariableOp<assignvariableop_2_decoder_decoder_block_18_conv2d_62_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_3AssignVariableOp:assignvariableop_3_decoder_decoder_block_18_conv2d_62_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_4AssignVariableOpHassignvariableop_4_decoder_decoder_block_18_batch_normalization_50_gammaIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:я
AssignVariableOp_5AssignVariableOpGassignvariableop_5_decoder_decoder_block_18_batch_normalization_50_betaIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_6AssignVariableOpNassignvariableop_6_decoder_decoder_block_18_batch_normalization_50_moving_meanIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_7AssignVariableOpRassignvariableop_7_decoder_decoder_block_18_batch_normalization_50_moving_varianceIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_8AssignVariableOp<assignvariableop_8_decoder_decoder_block_19_conv2d_63_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_9AssignVariableOp:assignvariableop_9_decoder_decoder_block_19_conv2d_63_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOpIassignvariableop_10_decoder_decoder_block_19_batch_normalization_51_gammaIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_11AssignVariableOpHassignvariableop_11_decoder_decoder_block_19_batch_normalization_51_betaIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_12AssignVariableOpOassignvariableop_12_decoder_decoder_block_19_batch_normalization_51_moving_meanIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_13AssignVariableOpSassignvariableop_13_decoder_decoder_block_19_batch_normalization_51_moving_varianceIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_14AssignVariableOp=assignvariableop_14_decoder_decoder_block_20_conv2d_64_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_15AssignVariableOp;assignvariableop_15_decoder_decoder_block_20_conv2d_64_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_16AssignVariableOpIassignvariableop_16_decoder_decoder_block_20_batch_normalization_52_gammaIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_17AssignVariableOpHassignvariableop_17_decoder_decoder_block_20_batch_normalization_52_betaIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_18AssignVariableOpOassignvariableop_18_decoder_decoder_block_20_batch_normalization_52_moving_meanIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_19AssignVariableOpSassignvariableop_19_decoder_decoder_block_20_batch_normalization_52_moving_varianceIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_20AssignVariableOp,assignvariableop_20_decoder_conv2d_65_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_21AssignVariableOp*assignvariableop_21_decoder_conv2d_65_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_22AssignVariableOp,assignvariableop_22_decoder_conv2d_66_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_23AssignVariableOp*assignvariableop_23_decoder_conv2d_66_biasIdentity_23:output:0"/device:CPU:0*&
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
ѕ
┬
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938605

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
ы
 
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1938437

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ф
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ј
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
у&
ы
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937008
input_tensorD
(conv2d_63_conv2d_readvariableop_resource:ђђ8
)conv2d_63_biasadd_readvariableop_resource:	ђ=
.batch_normalization_51_readvariableop_resource:	ђ?
0batch_normalization_51_readvariableop_1_resource:	ђN
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_51/AssignNewValueб'batch_normalization_51/AssignNewValue_1б6batch_normalization_51/FusedBatchNormV3/ReadVariableOpб8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_51/ReadVariableOpб'batch_normalization_51/ReadVariableOp_1б conv2d_63/BiasAdd/ReadVariableOpбconv2d_63/Conv2D/ReadVariableOpњ
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_63/Conv2DConv2Dinput_tensor'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0С
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_63/Relu:activations:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_51/AssignNewValueAssignVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource4batch_normalization_51/FusedBatchNormV3:batch_mean:07^batch_normalization_51/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_51/AssignNewValue_1AssignVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_51/FusedBatchNormV3:batch_variance:09^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ћ
IdentityIdentity+batch_normalization_51/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђБ
NoOpNoOp&^batch_normalization_51/AssignNewValue(^batch_normalization_51/AssignNewValue_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2R
'batch_normalization_51/AssignNewValue_1'batch_normalization_51/AssignNewValue_12N
%batch_normalization_51/AssignNewValue%batch_normalization_51/AssignNewValue2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
ў
к
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938543

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
 	
њ
2__inference_decoder_block_20_layer_call_fn_1938347
input_tensor"
unknown:ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937210Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
╝
а
+__inference_conv2d_65_layer_call_fn_1938406

inputs!
unknown:@ 
	unknown_0: 
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1937073Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Є

ў
2__inference_decoder_block_18_layer_call_fn_1938145
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1937132і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
й
Ъ
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938296
input_tensorD
(conv2d_63_conv2d_readvariableop_resource:ђђ8
)conv2d_63_biasadd_readvariableop_resource:	ђ=
.batch_normalization_51_readvariableop_resource:	ђ?
0batch_normalization_51_readvariableop_1_resource:	ђN
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_51/FusedBatchNormV3/ReadVariableOpб8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_51/ReadVariableOpб'batch_normalization_51/ReadVariableOp_1б conv2d_63/BiasAdd/ReadVariableOpбconv2d_63/Conv2D/ReadVariableOpњ
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_63/Conv2DConv2Dinput_tensor'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0о
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_63/Relu:activations:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ћ
IdentityIdentity+batch_normalization_51/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЛ
NoOpNoOp7^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
е
Ў
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938397
input_tensorC
(conv2d_64_conv2d_readvariableop_resource:ђ@7
)conv2d_64_biasadd_readvariableop_resource:@<
.batch_normalization_52_readvariableop_resource:@>
0batch_normalization_52_readvariableop_1_resource:@M
?batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб6batch_normalization_52/FusedBatchNormV3/ReadVariableOpб8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_52/ReadVariableOpб'batch_normalization_52/ReadVariableOp_1б conv2d_64/BiasAdd/ReadVariableOpбconv2d_64/Conv2D/ReadVariableOpЉ
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0┼
conv2d_64/Conv2DConv2Dinput_tensor'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @љ
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Х
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Л
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3conv2d_64/Relu:activations:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( ћ
IdentityIdentity+batch_normalization_52/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @Л
NoOpNoOp7^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_1!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
Б
i
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1936766

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
у&
ы
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938271
input_tensorD
(conv2d_63_conv2d_readvariableop_resource:ђђ8
)conv2d_63_biasadd_readvariableop_resource:	ђ=
.batch_normalization_51_readvariableop_resource:	ђ?
0batch_normalization_51_readvariableop_1_resource:	ђN
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_51/AssignNewValueб'batch_normalization_51/AssignNewValue_1б6batch_normalization_51/FusedBatchNormV3/ReadVariableOpб8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_51/ReadVariableOpб'batch_normalization_51/ReadVariableOp_1б conv2d_63/BiasAdd/ReadVariableOpбconv2d_63/Conv2D/ReadVariableOpњ
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_63/Conv2DConv2Dinput_tensor'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0С
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_63/Relu:activations:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_51/AssignNewValueAssignVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource4batch_normalization_51/FusedBatchNormV3:batch_mean:07^batch_normalization_51/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_51/AssignNewValue_1AssignVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_51/FusedBatchNormV3:batch_variance:09^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ћ
IdentityIdentity+batch_normalization_51/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђБ
NoOpNoOp&^batch_normalization_51/AssignNewValue(^batch_normalization_51/AssignNewValue_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2R
'batch_normalization_51/AssignNewValue_1'batch_normalization_51/AssignNewValue_12N
%batch_normalization_51/AssignNewValue%batch_normalization_51/AssignNewValue2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
б	
О
8__inference_batch_normalization_51_layer_call_fn_1938525

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
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
GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1936809і
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
ў
к
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1936708

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Є

ў
2__inference_decoder_block_19_layer_call_fn_1938246
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937171і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
Б
i
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1938313

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
ъ░
В
D__inference_decoder_layer_call_and_return_conditional_losses_1937972
embedding_input<
'dense_22_matmul_readvariableop_resource:ђђђ8
(dense_22_biasadd_readvariableop_resource:
ђђU
9decoder_block_18_conv2d_62_conv2d_readvariableop_resource:ђђI
:decoder_block_18_conv2d_62_biasadd_readvariableop_resource:	ђN
?decoder_block_18_batch_normalization_50_readvariableop_resource:	ђP
Adecoder_block_18_batch_normalization_50_readvariableop_1_resource:	ђ_
Pdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђa
Rdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђU
9decoder_block_19_conv2d_63_conv2d_readvariableop_resource:ђђI
:decoder_block_19_conv2d_63_biasadd_readvariableop_resource:	ђN
?decoder_block_19_batch_normalization_51_readvariableop_resource:	ђP
Adecoder_block_19_batch_normalization_51_readvariableop_1_resource:	ђ_
Pdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђa
Rdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђT
9decoder_block_20_conv2d_64_conv2d_readvariableop_resource:ђ@H
:decoder_block_20_conv2d_64_biasadd_readvariableop_resource:@M
?decoder_block_20_batch_normalization_52_readvariableop_resource:@O
Adecoder_block_20_batch_normalization_52_readvariableop_1_resource:@^
Pdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@`
Rdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_65_conv2d_readvariableop_resource:@ 7
)conv2d_65_biasadd_readvariableop_resource: B
(conv2d_66_conv2d_readvariableop_resource: 7
)conv2d_66_biasadd_readvariableop_resource:
identityѕб conv2d_65/BiasAdd/ReadVariableOpбconv2d_65/Conv2D/ReadVariableOpб conv2d_66/BiasAdd/ReadVariableOpбconv2d_66/Conv2D/ReadVariableOpб6decoder_block_18/batch_normalization_50/AssignNewValueб8decoder_block_18/batch_normalization_50/AssignNewValue_1бGdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpбIdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б6decoder_block_18/batch_normalization_50/ReadVariableOpб8decoder_block_18/batch_normalization_50/ReadVariableOp_1б1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOpб0decoder_block_18/conv2d_62/Conv2D/ReadVariableOpб6decoder_block_19/batch_normalization_51/AssignNewValueб8decoder_block_19/batch_normalization_51/AssignNewValue_1бGdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpбIdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б6decoder_block_19/batch_normalization_51/ReadVariableOpб8decoder_block_19/batch_normalization_51/ReadVariableOp_1б1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOpб0decoder_block_19/conv2d_63/Conv2D/ReadVariableOpб6decoder_block_20/batch_normalization_52/AssignNewValueб8decoder_block_20/batch_normalization_52/AssignNewValue_1бGdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpбIdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б6decoder_block_20/batch_normalization_52/ReadVariableOpб8decoder_block_20/batch_normalization_52/ReadVariableOp_1б1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOpб0decoder_block_20/conv2d_64/Conv2D/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpЅ
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0є
dense_22/MatMulMatMulembedding_input&dense_22/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђє
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0Њ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђd
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*)
_output_shapes
:         ђђf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ѓ
ReshapeReshapedense_22/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђg
up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ё
up_sampling2d_18/mulMulup_sampling2d_18/Const:output:0!up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:К
-up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborReshape:output:0up_sampling2d_18/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(┤
0decoder_block_18/conv2d_62/Conv2D/ReadVariableOpReadVariableOp9decoder_block_18_conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ѕ
!decoder_block_18/conv2d_62/Conv2DConv2D>up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:08decoder_block_18/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Е
1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_18_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"decoder_block_18/conv2d_62/BiasAddBiasAdd*decoder_block_18/conv2d_62/Conv2D:output:09decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЈ
decoder_block_18/conv2d_62/ReluRelu+decoder_block_18/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ│
6decoder_block_18/batch_normalization_50/ReadVariableOpReadVariableOp?decoder_block_18_batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8decoder_block_18/batch_normalization_50/ReadVariableOp_1ReadVariableOpAdecoder_block_18_batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Idecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0И
8decoder_block_18/batch_normalization_50/FusedBatchNormV3FusedBatchNormV3-decoder_block_18/conv2d_62/Relu:activations:0>decoder_block_18/batch_normalization_50/ReadVariableOp:value:0@decoder_block_18/batch_normalization_50/ReadVariableOp_1:value:0Odecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6decoder_block_18/batch_normalization_50/AssignNewValueAssignVariableOpPdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resourceEdecoder_block_18/batch_normalization_50/FusedBatchNormV3:batch_mean:0H^decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8decoder_block_18/batch_normalization_50/AssignNewValue_1AssignVariableOpRdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resourceIdecoder_block_18/batch_normalization_50/FusedBatchNormV3:batch_variance:0J^decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(g
up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ё
up_sampling2d_19/mulMulup_sampling2d_19/Const:output:0!up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:з
-up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighbor<decoder_block_18/batch_normalization_50/FusedBatchNormV3:y:0up_sampling2d_19/mul:z:0*
T0*0
_output_shapes
:           ђ*
half_pixel_centers(┤
0decoder_block_19/conv2d_63/Conv2D/ReadVariableOpReadVariableOp9decoder_block_19_conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ѕ
!decoder_block_19/conv2d_63/Conv2DConv2D>up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:08decoder_block_19/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Е
1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_19_conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"decoder_block_19/conv2d_63/BiasAddBiasAdd*decoder_block_19/conv2d_63/Conv2D:output:09decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђЈ
decoder_block_19/conv2d_63/ReluRelu+decoder_block_19/conv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ│
6decoder_block_19/batch_normalization_51/ReadVariableOpReadVariableOp?decoder_block_19_batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8decoder_block_19/batch_normalization_51/ReadVariableOp_1ReadVariableOpAdecoder_block_19_batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Idecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0И
8decoder_block_19/batch_normalization_51/FusedBatchNormV3FusedBatchNormV3-decoder_block_19/conv2d_63/Relu:activations:0>decoder_block_19/batch_normalization_51/ReadVariableOp:value:0@decoder_block_19/batch_normalization_51/ReadVariableOp_1:value:0Odecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6decoder_block_19/batch_normalization_51/AssignNewValueAssignVariableOpPdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resourceEdecoder_block_19/batch_normalization_51/FusedBatchNormV3:batch_mean:0H^decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8decoder_block_19/batch_normalization_51/AssignNewValue_1AssignVariableOpRdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resourceIdecoder_block_19/batch_normalization_51/FusedBatchNormV3:batch_variance:0J^decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(g
up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        i
up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ё
up_sampling2d_20/mulMulup_sampling2d_20/Const:output:0!up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:ш
-up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighbor<decoder_block_19/batch_normalization_51/FusedBatchNormV3:y:0up_sampling2d_20/mul:z:0*
T0*2
_output_shapes 
:         ђђђ*
half_pixel_centers(│
0decoder_block_20/conv2d_64/Conv2D/ReadVariableOpReadVariableOp9decoder_block_20_conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0Ѕ
!decoder_block_20/conv2d_64/Conv2DConv2D>up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:08decoder_block_20/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@*
paddingSAME*
strides
е
1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_20_conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
"decoder_block_20/conv2d_64/BiasAddBiasAdd*decoder_block_20/conv2d_64/Conv2D:output:09decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@љ
decoder_block_20/conv2d_64/ReluRelu+decoder_block_20/conv2d_64/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@▓
6decoder_block_20/batch_normalization_52/ReadVariableOpReadVariableOp?decoder_block_20_batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0Х
8decoder_block_20/batch_normalization_52/ReadVariableOp_1ReadVariableOpAdecoder_block_20_batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
Gdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0п
Idecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0х
8decoder_block_20/batch_normalization_52/FusedBatchNormV3FusedBatchNormV3-decoder_block_20/conv2d_64/Relu:activations:0>decoder_block_20/batch_normalization_52/ReadVariableOp:value:0@decoder_block_20/batch_normalization_52/ReadVariableOp_1:value:0Odecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6decoder_block_20/batch_normalization_52/AssignNewValueAssignVariableOpPdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resourceEdecoder_block_20/batch_normalization_52/FusedBatchNormV3:batch_mean:0H^decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8decoder_block_20/batch_normalization_52/AssignNewValue_1AssignVariableOpRdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resourceIdecoder_block_20/batch_normalization_52/FusedBatchNormV3:batch_variance:0J^decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(љ
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0т
conv2d_65/Conv2DConv2D<decoder_block_20/batch_normalization_52/FusedBatchNormV3:y:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ *
paddingSAME*
strides
є
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ n
conv2d_65/ReluReluconv2d_65/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ љ
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┼
conv2d_66/Conv2DConv2Dconv2d_65/Relu:activations:0'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
є
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђt
conv2d_66/SigmoidSigmoidconv2d_66/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђn
IdentityIdentityconv2d_66/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђ┬
NoOpNoOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOp7^decoder_block_18/batch_normalization_50/AssignNewValue9^decoder_block_18/batch_normalization_50/AssignNewValue_1H^decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpJ^decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_17^decoder_block_18/batch_normalization_50/ReadVariableOp9^decoder_block_18/batch_normalization_50/ReadVariableOp_12^decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp1^decoder_block_18/conv2d_62/Conv2D/ReadVariableOp7^decoder_block_19/batch_normalization_51/AssignNewValue9^decoder_block_19/batch_normalization_51/AssignNewValue_1H^decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpJ^decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_17^decoder_block_19/batch_normalization_51/ReadVariableOp9^decoder_block_19/batch_normalization_51/ReadVariableOp_12^decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp1^decoder_block_19/conv2d_63/Conv2D/ReadVariableOp7^decoder_block_20/batch_normalization_52/AssignNewValue9^decoder_block_20/batch_normalization_52/AssignNewValue_1H^decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpJ^decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_17^decoder_block_20/batch_normalization_52/ReadVariableOp9^decoder_block_20/batch_normalization_52/ReadVariableOp_12^decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp1^decoder_block_20/conv2d_64/Conv2D/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2t
8decoder_block_18/batch_normalization_50/AssignNewValue_18decoder_block_18/batch_normalization_50/AssignNewValue_12p
6decoder_block_18/batch_normalization_50/AssignNewValue6decoder_block_18/batch_normalization_50/AssignNewValue2ќ
Idecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12њ
Gdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpGdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_18/batch_normalization_50/ReadVariableOp_18decoder_block_18/batch_normalization_50/ReadVariableOp_12p
6decoder_block_18/batch_normalization_50/ReadVariableOp6decoder_block_18/batch_normalization_50/ReadVariableOp2f
1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp2d
0decoder_block_18/conv2d_62/Conv2D/ReadVariableOp0decoder_block_18/conv2d_62/Conv2D/ReadVariableOp2t
8decoder_block_19/batch_normalization_51/AssignNewValue_18decoder_block_19/batch_normalization_51/AssignNewValue_12p
6decoder_block_19/batch_normalization_51/AssignNewValue6decoder_block_19/batch_normalization_51/AssignNewValue2ќ
Idecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12њ
Gdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpGdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_19/batch_normalization_51/ReadVariableOp_18decoder_block_19/batch_normalization_51/ReadVariableOp_12p
6decoder_block_19/batch_normalization_51/ReadVariableOp6decoder_block_19/batch_normalization_51/ReadVariableOp2f
1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp2d
0decoder_block_19/conv2d_63/Conv2D/ReadVariableOp0decoder_block_19/conv2d_63/Conv2D/ReadVariableOp2t
8decoder_block_20/batch_normalization_52/AssignNewValue_18decoder_block_20/batch_normalization_52/AssignNewValue_12p
6decoder_block_20/batch_normalization_52/AssignNewValue6decoder_block_20/batch_normalization_52/AssignNewValue2ќ
Idecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12њ
Gdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpGdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_20/batch_normalization_52/ReadVariableOp_18decoder_block_20/batch_normalization_52/ReadVariableOp_12p
6decoder_block_20/batch_normalization_52/ReadVariableOp6decoder_block_20/batch_normalization_52/ReadVariableOp2f
1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp2d
0decoder_block_20/conv2d_64/Conv2D/ReadVariableOp0decoder_block_20/conv2d_64/Conv2D/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
Б
i
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1938212

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
§	
њ
2__inference_decoder_block_20_layer_call_fn_1938330
input_tensor"
unknown:ђ@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937048Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
я
б
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938561

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
ы
 
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1937090

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ф
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ј
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+                           w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
й
Ъ
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1937132
input_tensorD
(conv2d_62_conv2d_readvariableop_resource:ђђ8
)conv2d_62_biasadd_readvariableop_resource:	ђ=
.batch_normalization_50_readvariableop_resource:	ђ?
0batch_normalization_50_readvariableop_1_resource:	ђN
?batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_50/FusedBatchNormV3/ReadVariableOpб8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_50/ReadVariableOpб'batch_normalization_50/ReadVariableOp_1б conv2d_62/BiasAdd/ReadVariableOpбconv2d_62/Conv2D/ReadVariableOpњ
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_62/Conv2DConv2Dinput_tensor'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0о
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3conv2d_62/Relu:activations:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ћ
IdentityIdentity+batch_normalization_50/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЛ
NoOpNoOp7^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
Б
i
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1938111

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
┴
N
2__inference_up_sampling2d_20_layer_call_fn_1938301

inputs
identityЯ
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1936849Ѓ
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
Н
џ
)__inference_decoder_layer_call_fn_1937352
input_1
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ%

unknown_13:ђ@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: $

unknown_21: 

unknown_22:
identityѕбStatefulPartitionedCallъ
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
-:+                           *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1937301Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
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
█
џ
)__inference_decoder_layer_call_fn_1937468
input_1
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ%

unknown_13:ђ@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: $

unknown_21: 

unknown_22:
identityѕбStatefulPartitionedCallц
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
-:+                           *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1937417Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
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
б	
О
8__inference_batch_normalization_50_layer_call_fn_1938463

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1936726і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
┼6
И

D__inference_decoder_layer_call_and_return_conditional_losses_1937097
input_1%
dense_22_1936935:ђђђ 
dense_22_1936937:
ђђ4
decoder_block_18_1936969:ђђ'
decoder_block_18_1936971:	ђ'
decoder_block_18_1936973:	ђ'
decoder_block_18_1936975:	ђ'
decoder_block_18_1936977:	ђ'
decoder_block_18_1936979:	ђ4
decoder_block_19_1937009:ђђ'
decoder_block_19_1937011:	ђ'
decoder_block_19_1937013:	ђ'
decoder_block_19_1937015:	ђ'
decoder_block_19_1937017:	ђ'
decoder_block_19_1937019:	ђ3
decoder_block_20_1937049:ђ@&
decoder_block_20_1937051:@&
decoder_block_20_1937053:@&
decoder_block_20_1937055:@&
decoder_block_20_1937057:@&
decoder_block_20_1937059:@+
conv2d_65_1937074:@ 
conv2d_65_1937076: +
conv2d_66_1937091: 
conv2d_66_1937093:
identityѕб!conv2d_65/StatefulPartitionedCallб!conv2d_66/StatefulPartitionedCallб(decoder_block_18/StatefulPartitionedCallб(decoder_block_19/StatefulPartitionedCallб(decoder_block_20/StatefulPartitionedCallб dense_22/StatefulPartitionedCallч
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_22_1936935dense_22_1936937*
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1936934f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             љ
ReshapeReshape)dense_22/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђз
 up_sampling2d_18/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1936683─
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_18/PartitionedCall:output:0decoder_block_18_1936969decoder_block_18_1936971decoder_block_18_1936973decoder_block_18_1936975decoder_block_18_1936977decoder_block_18_1936979*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1936968ћ
 up_sampling2d_19/PartitionedCallPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1936766─
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_19/PartitionedCall:output:0decoder_block_19_1937009decoder_block_19_1937011decoder_block_19_1937013decoder_block_19_1937015decoder_block_19_1937017decoder_block_19_1937019*
Tin
	2*
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937008ћ
 up_sampling2d_20/PartitionedCallPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1936849├
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_20/PartitionedCall:output:0decoder_block_20_1937049decoder_block_20_1937051decoder_block_20_1937053decoder_block_20_1937055decoder_block_20_1937057decoder_block_20_1937059*
Tin
	2*
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937048┴
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_65_1937074conv2d_65_1937076*
Tin
2*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1937073║
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_1937091conv2d_66_1937093*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1937090Њ
IdentityIdentity*conv2d_66/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ▓
NoOpNoOp"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
П6
└

D__inference_decoder_layer_call_and_return_conditional_losses_1937301
embedding_input%
dense_22_1937241:ђђђ 
dense_22_1937243:
ђђ4
decoder_block_18_1937249:ђђ'
decoder_block_18_1937251:	ђ'
decoder_block_18_1937253:	ђ'
decoder_block_18_1937255:	ђ'
decoder_block_18_1937257:	ђ'
decoder_block_18_1937259:	ђ4
decoder_block_19_1937263:ђђ'
decoder_block_19_1937265:	ђ'
decoder_block_19_1937267:	ђ'
decoder_block_19_1937269:	ђ'
decoder_block_19_1937271:	ђ'
decoder_block_19_1937273:	ђ3
decoder_block_20_1937277:ђ@&
decoder_block_20_1937279:@&
decoder_block_20_1937281:@&
decoder_block_20_1937283:@&
decoder_block_20_1937285:@&
decoder_block_20_1937287:@+
conv2d_65_1937290:@ 
conv2d_65_1937292: +
conv2d_66_1937295: 
conv2d_66_1937297:
identityѕб!conv2d_65/StatefulPartitionedCallб!conv2d_66/StatefulPartitionedCallб(decoder_block_18/StatefulPartitionedCallб(decoder_block_19/StatefulPartitionedCallб(decoder_block_20/StatefulPartitionedCallб dense_22/StatefulPartitionedCallЃ
 dense_22/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_22_1937241dense_22_1937243*
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1936934f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             љ
ReshapeReshape)dense_22/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђз
 up_sampling2d_18/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1936683─
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_18/PartitionedCall:output:0decoder_block_18_1937249decoder_block_18_1937251decoder_block_18_1937253decoder_block_18_1937255decoder_block_18_1937257decoder_block_18_1937259*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1936968ћ
 up_sampling2d_19/PartitionedCallPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1936766─
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_19/PartitionedCall:output:0decoder_block_19_1937263decoder_block_19_1937265decoder_block_19_1937267decoder_block_19_1937269decoder_block_19_1937271decoder_block_19_1937273*
Tin
	2*
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937008ћ
 up_sampling2d_20/PartitionedCallPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1936849├
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_20/PartitionedCall:output:0decoder_block_20_1937277decoder_block_20_1937279decoder_block_20_1937281decoder_block_20_1937283decoder_block_20_1937285decoder_block_20_1937287*
Tin
	2*
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937048┴
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_65_1937290conv2d_65_1937292*
Tin
2*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1937073║
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_1937295conv2d_66_1937297*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1937090Њ
IdentityIdentity*conv2d_66/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ▓
NoOpNoOp"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
ў
к
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938481

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ы
 
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1937073

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ф
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ј
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
з
б
)__inference_decoder_layer_call_fn_1937870
embedding_input
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ%

unknown_13:ђ@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: $

unknown_21: 

unknown_22:
identityѕбStatefulPartitionedCallг
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
-:+                           *:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1937417Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
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
Ё

ў
2__inference_decoder_block_18_layer_call_fn_1938128
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1936968і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
ў	
М
8__inference_batch_normalization_52_layer_call_fn_1938574

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallЪ
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
GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1936874Ѕ
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
я
б
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1936726

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
­Њ
љ
D__inference_decoder_layer_call_and_return_conditional_losses_1938074
embedding_input<
'dense_22_matmul_readvariableop_resource:ђђђ8
(dense_22_biasadd_readvariableop_resource:
ђђU
9decoder_block_18_conv2d_62_conv2d_readvariableop_resource:ђђI
:decoder_block_18_conv2d_62_biasadd_readvariableop_resource:	ђN
?decoder_block_18_batch_normalization_50_readvariableop_resource:	ђP
Adecoder_block_18_batch_normalization_50_readvariableop_1_resource:	ђ_
Pdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђa
Rdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђU
9decoder_block_19_conv2d_63_conv2d_readvariableop_resource:ђђI
:decoder_block_19_conv2d_63_biasadd_readvariableop_resource:	ђN
?decoder_block_19_batch_normalization_51_readvariableop_resource:	ђP
Adecoder_block_19_batch_normalization_51_readvariableop_1_resource:	ђ_
Pdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђa
Rdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђT
9decoder_block_20_conv2d_64_conv2d_readvariableop_resource:ђ@H
:decoder_block_20_conv2d_64_biasadd_readvariableop_resource:@M
?decoder_block_20_batch_normalization_52_readvariableop_resource:@O
Adecoder_block_20_batch_normalization_52_readvariableop_1_resource:@^
Pdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@`
Rdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@B
(conv2d_65_conv2d_readvariableop_resource:@ 7
)conv2d_65_biasadd_readvariableop_resource: B
(conv2d_66_conv2d_readvariableop_resource: 7
)conv2d_66_biasadd_readvariableop_resource:
identityѕб conv2d_65/BiasAdd/ReadVariableOpбconv2d_65/Conv2D/ReadVariableOpб conv2d_66/BiasAdd/ReadVariableOpбconv2d_66/Conv2D/ReadVariableOpбGdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpбIdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б6decoder_block_18/batch_normalization_50/ReadVariableOpб8decoder_block_18/batch_normalization_50/ReadVariableOp_1б1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOpб0decoder_block_18/conv2d_62/Conv2D/ReadVariableOpбGdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpбIdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б6decoder_block_19/batch_normalization_51/ReadVariableOpб8decoder_block_19/batch_normalization_51/ReadVariableOp_1б1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOpб0decoder_block_19/conv2d_63/Conv2D/ReadVariableOpбGdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpбIdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б6decoder_block_20/batch_normalization_52/ReadVariableOpб8decoder_block_20/batch_normalization_52/ReadVariableOp_1б1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOpб0decoder_block_20/conv2d_64/Conv2D/ReadVariableOpбdense_22/BiasAdd/ReadVariableOpбdense_22/MatMul/ReadVariableOpЅ
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*!
_output_shapes
:ђђђ*
dtype0є
dense_22/MatMulMatMulembedding_input&dense_22/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђє
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes

:ђђ*
dtype0Њ
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:         ђђd
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*)
_output_shapes
:         ђђf
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ѓ
ReshapeReshapedense_22/Relu:activations:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђg
up_sampling2d_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ё
up_sampling2d_18/mulMulup_sampling2d_18/Const:output:0!up_sampling2d_18/Const_1:output:0*
T0*
_output_shapes
:К
-up_sampling2d_18/resize/ResizeNearestNeighborResizeNearestNeighborReshape:output:0up_sampling2d_18/mul:z:0*
T0*0
_output_shapes
:         ђ*
half_pixel_centers(┤
0decoder_block_18/conv2d_62/Conv2D/ReadVariableOpReadVariableOp9decoder_block_18_conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ѕ
!decoder_block_18/conv2d_62/Conv2DConv2D>up_sampling2d_18/resize/ResizeNearestNeighbor:resized_images:08decoder_block_18/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Е
1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_18_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"decoder_block_18/conv2d_62/BiasAddBiasAdd*decoder_block_18/conv2d_62/Conv2D:output:09decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЈ
decoder_block_18/conv2d_62/ReluRelu+decoder_block_18/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ│
6decoder_block_18/batch_normalization_50/ReadVariableOpReadVariableOp?decoder_block_18_batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8decoder_block_18/batch_normalization_50/ReadVariableOp_1ReadVariableOpAdecoder_block_18_batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Idecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_18_batch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0ф
8decoder_block_18/batch_normalization_50/FusedBatchNormV3FusedBatchNormV3-decoder_block_18/conv2d_62/Relu:activations:0>decoder_block_18/batch_normalization_50/ReadVariableOp:value:0@decoder_block_18/batch_normalization_50/ReadVariableOp_1:value:0Odecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( g
up_sampling2d_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"      i
up_sampling2d_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ё
up_sampling2d_19/mulMulup_sampling2d_19/Const:output:0!up_sampling2d_19/Const_1:output:0*
T0*
_output_shapes
:з
-up_sampling2d_19/resize/ResizeNearestNeighborResizeNearestNeighbor<decoder_block_18/batch_normalization_50/FusedBatchNormV3:y:0up_sampling2d_19/mul:z:0*
T0*0
_output_shapes
:           ђ*
half_pixel_centers(┤
0decoder_block_19/conv2d_63/Conv2D/ReadVariableOpReadVariableOp9decoder_block_19_conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ѕ
!decoder_block_19/conv2d_63/Conv2DConv2D>up_sampling2d_19/resize/ResizeNearestNeighbor:resized_images:08decoder_block_19/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Е
1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_19_conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"decoder_block_19/conv2d_63/BiasAddBiasAdd*decoder_block_19/conv2d_63/Conv2D:output:09decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђЈ
decoder_block_19/conv2d_63/ReluRelu+decoder_block_19/conv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ│
6decoder_block_19/batch_normalization_51/ReadVariableOpReadVariableOp?decoder_block_19_batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8decoder_block_19/batch_normalization_51/ReadVariableOp_1ReadVariableOpAdecoder_block_19_batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Idecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_19_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0ф
8decoder_block_19/batch_normalization_51/FusedBatchNormV3FusedBatchNormV3-decoder_block_19/conv2d_63/Relu:activations:0>decoder_block_19/batch_normalization_51/ReadVariableOp:value:0@decoder_block_19/batch_normalization_51/ReadVariableOp_1:value:0Odecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( g
up_sampling2d_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"        i
up_sampling2d_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      ё
up_sampling2d_20/mulMulup_sampling2d_20/Const:output:0!up_sampling2d_20/Const_1:output:0*
T0*
_output_shapes
:ш
-up_sampling2d_20/resize/ResizeNearestNeighborResizeNearestNeighbor<decoder_block_19/batch_normalization_51/FusedBatchNormV3:y:0up_sampling2d_20/mul:z:0*
T0*2
_output_shapes 
:         ђђђ*
half_pixel_centers(│
0decoder_block_20/conv2d_64/Conv2D/ReadVariableOpReadVariableOp9decoder_block_20_conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0Ѕ
!decoder_block_20/conv2d_64/Conv2DConv2D>up_sampling2d_20/resize/ResizeNearestNeighbor:resized_images:08decoder_block_20/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@*
paddingSAME*
strides
е
1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOpReadVariableOp:decoder_block_20_conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0л
"decoder_block_20/conv2d_64/BiasAddBiasAdd*decoder_block_20/conv2d_64/Conv2D:output:09decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ@љ
decoder_block_20/conv2d_64/ReluRelu+decoder_block_20/conv2d_64/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ@▓
6decoder_block_20/batch_normalization_52/ReadVariableOpReadVariableOp?decoder_block_20_batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0Х
8decoder_block_20/batch_normalization_52/ReadVariableOp_1ReadVariableOpAdecoder_block_20_batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0н
Gdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOpPdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0п
Idecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRdecoder_block_20_batch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Д
8decoder_block_20/batch_normalization_52/FusedBatchNormV3FusedBatchNormV3-decoder_block_20/conv2d_64/Relu:activations:0>decoder_block_20/batch_normalization_52/ReadVariableOp:value:0@decoder_block_20/batch_normalization_52/ReadVariableOp_1:value:0Odecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0Qdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ђђ@:@:@:@:@:*
epsilon%oЃ:*
is_training( љ
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0т
conv2d_65/Conv2DConv2D<decoder_block_20/batch_normalization_52/FusedBatchNormV3:y:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ *
paddingSAME*
strides
є
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ю
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ n
conv2d_65/ReluReluconv2d_65/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђ љ
conv2d_66/Conv2D/ReadVariableOpReadVariableOp(conv2d_66_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0┼
conv2d_66/Conv2DConv2Dconv2d_65/Relu:activations:0'conv2d_66/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђ*
paddingSAME*
strides
є
 conv2d_66/BiasAdd/ReadVariableOpReadVariableOp)conv2d_66_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ю
conv2d_66/BiasAddBiasAddconv2d_66/Conv2D:output:0(conv2d_66/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ђђt
conv2d_66/SigmoidSigmoidconv2d_66/BiasAdd:output:0*
T0*1
_output_shapes
:         ђђn
IdentityIdentityconv2d_66/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:         ђђТ

NoOpNoOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp!^conv2d_66/BiasAdd/ReadVariableOp ^conv2d_66/Conv2D/ReadVariableOpH^decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpJ^decoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_17^decoder_block_18/batch_normalization_50/ReadVariableOp9^decoder_block_18/batch_normalization_50/ReadVariableOp_12^decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp1^decoder_block_18/conv2d_62/Conv2D/ReadVariableOpH^decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpJ^decoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_17^decoder_block_19/batch_normalization_51/ReadVariableOp9^decoder_block_19/batch_normalization_51/ReadVariableOp_12^decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp1^decoder_block_19/conv2d_63/Conv2D/ReadVariableOpH^decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpJ^decoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_17^decoder_block_20/batch_normalization_52/ReadVariableOp9^decoder_block_20/batch_normalization_52/ReadVariableOp_12^decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp1^decoder_block_20/conv2d_64/Conv2D/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2D
 conv2d_66/BiasAdd/ReadVariableOp conv2d_66/BiasAdd/ReadVariableOp2B
conv2d_66/Conv2D/ReadVariableOpconv2d_66/Conv2D/ReadVariableOp2ќ
Idecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12њ
Gdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOpGdecoder_block_18/batch_normalization_50/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_18/batch_normalization_50/ReadVariableOp_18decoder_block_18/batch_normalization_50/ReadVariableOp_12p
6decoder_block_18/batch_normalization_50/ReadVariableOp6decoder_block_18/batch_normalization_50/ReadVariableOp2f
1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp1decoder_block_18/conv2d_62/BiasAdd/ReadVariableOp2d
0decoder_block_18/conv2d_62/Conv2D/ReadVariableOp0decoder_block_18/conv2d_62/Conv2D/ReadVariableOp2ќ
Idecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12њ
Gdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOpGdecoder_block_19/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_19/batch_normalization_51/ReadVariableOp_18decoder_block_19/batch_normalization_51/ReadVariableOp_12p
6decoder_block_19/batch_normalization_51/ReadVariableOp6decoder_block_19/batch_normalization_51/ReadVariableOp2f
1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp1decoder_block_19/conv2d_63/BiasAdd/ReadVariableOp2d
0decoder_block_19/conv2d_63/Conv2D/ReadVariableOp0decoder_block_19/conv2d_63/Conv2D/ReadVariableOp2ќ
Idecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1Idecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12њ
Gdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOpGdecoder_block_20/batch_normalization_52/FusedBatchNormV3/ReadVariableOp2t
8decoder_block_20/batch_normalization_52/ReadVariableOp_18decoder_block_20/batch_normalization_52/ReadVariableOp_12p
6decoder_block_20/batch_normalization_52/ReadVariableOp6decoder_block_20/batch_normalization_52/ReadVariableOp2f
1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp1decoder_block_20/conv2d_64/BiasAdd/ReadVariableOp2d
0decoder_block_20/conv2d_64/Conv2D/ReadVariableOp0decoder_block_20/conv2d_64/Conv2D/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
с6
└

D__inference_decoder_layer_call_and_return_conditional_losses_1937417
embedding_input%
dense_22_1937357:ђђђ 
dense_22_1937359:
ђђ4
decoder_block_18_1937365:ђђ'
decoder_block_18_1937367:	ђ'
decoder_block_18_1937369:	ђ'
decoder_block_18_1937371:	ђ'
decoder_block_18_1937373:	ђ'
decoder_block_18_1937375:	ђ4
decoder_block_19_1937379:ђђ'
decoder_block_19_1937381:	ђ'
decoder_block_19_1937383:	ђ'
decoder_block_19_1937385:	ђ'
decoder_block_19_1937387:	ђ'
decoder_block_19_1937389:	ђ3
decoder_block_20_1937393:ђ@&
decoder_block_20_1937395:@&
decoder_block_20_1937397:@&
decoder_block_20_1937399:@&
decoder_block_20_1937401:@&
decoder_block_20_1937403:@+
conv2d_65_1937406:@ 
conv2d_65_1937408: +
conv2d_66_1937411: 
conv2d_66_1937413:
identityѕб!conv2d_65/StatefulPartitionedCallб!conv2d_66/StatefulPartitionedCallб(decoder_block_18/StatefulPartitionedCallб(decoder_block_19/StatefulPartitionedCallб(decoder_block_20/StatefulPartitionedCallб dense_22/StatefulPartitionedCallЃ
 dense_22/StatefulPartitionedCallStatefulPartitionedCallembedding_inputdense_22_1937357dense_22_1937359*
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1936934f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             љ
ReshapeReshape)dense_22/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђз
 up_sampling2d_18/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1936683к
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_18/PartitionedCall:output:0decoder_block_18_1937365decoder_block_18_1937367decoder_block_18_1937369decoder_block_18_1937371decoder_block_18_1937373decoder_block_18_1937375*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1937132ћ
 up_sampling2d_19/PartitionedCallPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1936766к
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_19/PartitionedCall:output:0decoder_block_19_1937379decoder_block_19_1937381decoder_block_19_1937383decoder_block_19_1937385decoder_block_19_1937387decoder_block_19_1937389*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937171ћ
 up_sampling2d_20/PartitionedCallPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1936849┼
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_20/PartitionedCall:output:0decoder_block_20_1937393decoder_block_20_1937395decoder_block_20_1937397decoder_block_20_1937399decoder_block_20_1937401decoder_block_20_1937403*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937210┴
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_65_1937406conv2d_65_1937408*
Tin
2*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1937073║
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_1937411conv2d_66_1937413*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1937090Њ
IdentityIdentity*conv2d_66/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ▓
NoOpNoOp"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:Y U
(
_output_shapes
:         ђ
)
_user_specified_nameembedding_input
╦6
И

D__inference_decoder_layer_call_and_return_conditional_losses_1937235
input_1%
dense_22_1937100:ђђђ 
dense_22_1937102:
ђђ4
decoder_block_18_1937133:ђђ'
decoder_block_18_1937135:	ђ'
decoder_block_18_1937137:	ђ'
decoder_block_18_1937139:	ђ'
decoder_block_18_1937141:	ђ'
decoder_block_18_1937143:	ђ4
decoder_block_19_1937172:ђђ'
decoder_block_19_1937174:	ђ'
decoder_block_19_1937176:	ђ'
decoder_block_19_1937178:	ђ'
decoder_block_19_1937180:	ђ'
decoder_block_19_1937182:	ђ3
decoder_block_20_1937211:ђ@&
decoder_block_20_1937213:@&
decoder_block_20_1937215:@&
decoder_block_20_1937217:@&
decoder_block_20_1937219:@&
decoder_block_20_1937221:@+
conv2d_65_1937224:@ 
conv2d_65_1937226: +
conv2d_66_1937229: 
conv2d_66_1937231:
identityѕб!conv2d_65/StatefulPartitionedCallб!conv2d_66/StatefulPartitionedCallб(decoder_block_18/StatefulPartitionedCallб(decoder_block_19/StatefulPartitionedCallб(decoder_block_20/StatefulPartitionedCallб dense_22/StatefulPartitionedCallч
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_22_1937100dense_22_1937102*
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1936934f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             љ
ReshapeReshape)dense_22/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*0
_output_shapes
:         ђз
 up_sampling2d_18/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1936683к
(decoder_block_18/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_18/PartitionedCall:output:0decoder_block_18_1937133decoder_block_18_1937135decoder_block_18_1937137decoder_block_18_1937139decoder_block_18_1937141decoder_block_18_1937143*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1937132ћ
 up_sampling2d_19/PartitionedCallPartitionedCall1decoder_block_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1936766к
(decoder_block_19/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_19/PartitionedCall:output:0decoder_block_19_1937172decoder_block_19_1937174decoder_block_19_1937176decoder_block_19_1937178decoder_block_19_1937180decoder_block_19_1937182*
Tin
	2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937171ћ
 up_sampling2d_20/PartitionedCallPartitionedCall1decoder_block_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1936849┼
(decoder_block_20/StatefulPartitionedCallStatefulPartitionedCall)up_sampling2d_20/PartitionedCall:output:0decoder_block_20_1937211decoder_block_20_1937213decoder_block_20_1937215decoder_block_20_1937217decoder_block_20_1937219decoder_block_20_1937221*
Tin
	2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937210┴
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall1decoder_block_20/StatefulPartitionedCall:output:0conv2d_65_1937224conv2d_65_1937226*
Tin
2*
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
GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1937073║
!conv2d_66/StatefulPartitionedCallStatefulPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0conv2d_66_1937229conv2d_66_1937231*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1937090Њ
IdentityIdentity*conv2d_66/StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           ▓
NoOpNoOp"^conv2d_65/StatefulPartitionedCall"^conv2d_66/StatefulPartitionedCall)^decoder_block_18/StatefulPartitionedCall)^decoder_block_19/StatefulPartitionedCall)^decoder_block_20/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:         ђ: : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2F
!conv2d_66/StatefulPartitionedCall!conv2d_66/StatefulPartitionedCall2T
(decoder_block_18/StatefulPartitionedCall(decoder_block_18/StatefulPartitionedCall2T
(decoder_block_19/StatefulPartitionedCall(decoder_block_19/StatefulPartitionedCall2T
(decoder_block_20/StatefulPartitionedCall(decoder_block_20/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall:Q M
(
_output_shapes
:         ђ
!
_user_specified_name	input_1
е
Ў
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937210
input_tensorC
(conv2d_64_conv2d_readvariableop_resource:ђ@7
)conv2d_64_biasadd_readvariableop_resource:@<
.batch_normalization_52_readvariableop_resource:@>
0batch_normalization_52_readvariableop_1_resource:@M
?batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб6batch_normalization_52/FusedBatchNormV3/ReadVariableOpб8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_52/ReadVariableOpб'batch_normalization_52/ReadVariableOp_1б conv2d_64/BiasAdd/ReadVariableOpбconv2d_64/Conv2D/ReadVariableOpЉ
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0┼
conv2d_64/Conv2DConv2Dinput_tensor'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @љ
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Х
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Л
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3conv2d_64/Relu:activations:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( ћ
IdentityIdentity+batch_normalization_52/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @Л
NoOpNoOp7^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_1!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
я
б
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938499

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Б
i
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1936683

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
м&
в
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1937048
input_tensorC
(conv2d_64_conv2d_readvariableop_resource:ђ@7
)conv2d_64_biasadd_readvariableop_resource:@<
.batch_normalization_52_readvariableop_resource:@>
0batch_normalization_52_readvariableop_1_resource:@M
?batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб%batch_normalization_52/AssignNewValueб'batch_normalization_52/AssignNewValue_1б6batch_normalization_52/FusedBatchNormV3/ReadVariableOpб8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_52/ReadVariableOpб'batch_normalization_52/ReadVariableOp_1б conv2d_64/BiasAdd/ReadVariableOpбconv2d_64/Conv2D/ReadVariableOpЉ
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0┼
conv2d_64/Conv2DConv2Dinput_tensor'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @љ
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Х
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▀
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3conv2d_64/Relu:activations:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_52/AssignNewValueAssignVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource4batch_normalization_52/FusedBatchNormV3:batch_mean:07^batch_normalization_52/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_52/AssignNewValue_1AssignVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_52/FusedBatchNormV3:batch_variance:09^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ћ
IdentityIdentity+batch_normalization_52/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @Б
NoOpNoOp&^batch_normalization_52/AssignNewValue(^batch_normalization_52/AssignNewValue_17^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_1!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2R
'batch_normalization_52/AssignNewValue_1'batch_normalization_52/AssignNewValue_12N
%batch_normalization_52/AssignNewValue%batch_normalization_52/AssignNewValue2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
џ	
М
8__inference_batch_normalization_52_layer_call_fn_1938587

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallА
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
GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1936892Ѕ
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
ь
б
)__inference_decoder_layer_call_fn_1937817
embedding_input
unknown:ђђђ
	unknown_0:
ђђ%
	unknown_1:ђђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
	unknown_5:	ђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:	ђ%

unknown_13:ђ@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:@$

unknown_19:@ 

unknown_20: $

unknown_21: 

unknown_22:
identityѕбStatefulPartitionedCallд
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
-:+                           *4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_1937301Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
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
й
Ъ
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1937171
input_tensorD
(conv2d_63_conv2d_readvariableop_resource:ђђ8
)conv2d_63_biasadd_readvariableop_resource:	ђ=
.batch_normalization_51_readvariableop_resource:	ђ?
0batch_normalization_51_readvariableop_1_resource:	ђN
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_51/FusedBatchNormV3/ReadVariableOpб8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_51/ReadVariableOpб'batch_normalization_51/ReadVariableOp_1б conv2d_63/BiasAdd/ReadVariableOpбconv2d_63/Conv2D/ReadVariableOpњ
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_63/Conv2DConv2Dinput_tensor'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0о
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_63/Relu:activations:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ћ
IdentityIdentity+batch_normalization_51/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЛ
NoOpNoOp7^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
┴
N
2__inference_up_sampling2d_19_layer_call_fn_1938200

inputs
identityЯ
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
GPU2 *0J 8ѓ *V
fQRO
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1936766Ѓ
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
у&
ы
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938170
input_tensorD
(conv2d_62_conv2d_readvariableop_resource:ђђ8
)conv2d_62_biasadd_readvariableop_resource:	ђ=
.batch_normalization_50_readvariableop_resource:	ђ?
0batch_normalization_50_readvariableop_1_resource:	ђN
?batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_50/AssignNewValueб'batch_normalization_50/AssignNewValue_1б6batch_normalization_50/FusedBatchNormV3/ReadVariableOpб8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_50/ReadVariableOpб'batch_normalization_50/ReadVariableOp_1б conv2d_62/BiasAdd/ReadVariableOpбconv2d_62/Conv2D/ReadVariableOpњ
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_62/Conv2DConv2Dinput_tensor'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0С
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3conv2d_62/Relu:activations:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_50/AssignNewValueAssignVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource4batch_normalization_50/FusedBatchNormV3:batch_mean:07^batch_normalization_50/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_50/AssignNewValue_1AssignVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_50/FusedBatchNormV3:batch_variance:09^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ћ
IdentityIdentity+batch_normalization_50/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђБ
NoOpNoOp&^batch_normalization_50/AssignNewValue(^batch_normalization_50/AssignNewValue_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2R
'batch_normalization_50/AssignNewValue_1'batch_normalization_50/AssignNewValue_12N
%batch_normalization_50/AssignNewValue%batch_normalization_50/AssignNewValue2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
╬
ъ
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1936892

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
░

ч
E__inference_dense_22_layer_call_and_return_conditional_losses_1936934

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
у&
ы
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1936968
input_tensorD
(conv2d_62_conv2d_readvariableop_resource:ђђ8
)conv2d_62_biasadd_readvariableop_resource:	ђ=
.batch_normalization_50_readvariableop_resource:	ђ?
0batch_normalization_50_readvariableop_1_resource:	ђN
?batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_50/AssignNewValueб'batch_normalization_50/AssignNewValue_1б6batch_normalization_50/FusedBatchNormV3/ReadVariableOpб8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_50/ReadVariableOpб'batch_normalization_50/ReadVariableOp_1б conv2d_62/BiasAdd/ReadVariableOpбconv2d_62/Conv2D/ReadVariableOpњ
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_62/Conv2DConv2Dinput_tensor'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0С
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3conv2d_62/Relu:activations:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_50/AssignNewValueAssignVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource4batch_normalization_50/FusedBatchNormV3:batch_mean:07^batch_normalization_50/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_50/AssignNewValue_1AssignVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_50/FusedBatchNormV3:batch_variance:09^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ћ
IdentityIdentity+batch_normalization_50/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђБ
NoOpNoOp&^batch_normalization_50/AssignNewValue(^batch_normalization_50/AssignNewValue_17^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2R
'batch_normalization_50/AssignNewValue_1'batch_normalization_50/AssignNewValue_12N
%batch_normalization_50/AssignNewValue%batch_normalization_50/AssignNewValue2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
а	
О
8__inference_batch_normalization_51_layer_call_fn_1938512

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
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
GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1936791і
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
╝
а
+__inference_conv2d_66_layer_call_fn_1938426

inputs!
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1937090Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                            : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ы
 
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1938417

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0Ф
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ј
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                            w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
м&
в
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938372
input_tensorC
(conv2d_64_conv2d_readvariableop_resource:ђ@7
)conv2d_64_biasadd_readvariableop_resource:@<
.batch_normalization_52_readvariableop_resource:@>
0batch_normalization_52_readvariableop_1_resource:@M
?batch_normalization_52_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб%batch_normalization_52/AssignNewValueб'batch_normalization_52/AssignNewValue_1б6batch_normalization_52/FusedBatchNormV3/ReadVariableOpб8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_52/ReadVariableOpб'batch_normalization_52/ReadVariableOp_1б conv2d_64/BiasAdd/ReadVariableOpбconv2d_64/Conv2D/ReadVariableOpЉ
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*'
_output_shapes
:ђ@*
dtype0┼
conv2d_64/Conv2DConv2Dinput_tensor'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @*
paddingSAME*
strides
є
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @~
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @љ
%batch_normalization_52/ReadVariableOpReadVariableOp.batch_normalization_52_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
'batch_normalization_52/ReadVariableOp_1ReadVariableOp0batch_normalization_52_readvariableop_1_resource*
_output_shapes
:@*
dtype0▓
6batch_normalization_52/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Х
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▀
'batch_normalization_52/FusedBatchNormV3FusedBatchNormV3conv2d_64/Relu:activations:0-batch_normalization_52/ReadVariableOp:value:0/batch_normalization_52/ReadVariableOp_1:value:0>batch_normalization_52/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_52/AssignNewValueAssignVariableOp?batch_normalization_52_fusedbatchnormv3_readvariableop_resource4batch_normalization_52/FusedBatchNormV3:batch_mean:07^batch_normalization_52/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_52/AssignNewValue_1AssignVariableOpAbatch_normalization_52_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_52/FusedBatchNormV3:batch_variance:09^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ћ
IdentityIdentity+batch_normalization_52/FusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @Б
NoOpNoOp&^batch_normalization_52/AssignNewValue(^batch_normalization_52/AssignNewValue_17^batch_normalization_52/FusedBatchNormV3/ReadVariableOp9^batch_normalization_52/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_52/ReadVariableOp(^batch_normalization_52/ReadVariableOp_1!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2R
'batch_normalization_52/AssignNewValue_1'batch_normalization_52/AssignNewValue_12N
%batch_normalization_52/AssignNewValue%batch_normalization_52/AssignNewValue2t
8batch_normalization_52/FusedBatchNormV3/ReadVariableOp_18batch_normalization_52/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_52/FusedBatchNormV3/ReadVariableOp6batch_normalization_52/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_52/ReadVariableOp_1'batch_normalization_52/ReadVariableOp_12N
%batch_normalization_52/ReadVariableOp%batch_normalization_52/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor
н
ю
*__inference_dense_22_layer_call_fn_1938083

inputs
unknown:ђђђ
	unknown_0:
ђђ
identityѕбStatefulPartitionedCallр
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
GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_1936934q
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
╬
ъ
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938623

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
Б
i
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1936849

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
а	
О
8__inference_batch_normalization_50_layer_call_fn_1938450

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1936708і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ў
к
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1936791

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
░

ч
E__inference_dense_22_layer_call_and_return_conditional_losses_1938094

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
я
б
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1936809

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
й
Ъ
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938195
input_tensorD
(conv2d_62_conv2d_readvariableop_resource:ђђ8
)conv2d_62_biasadd_readvariableop_resource:	ђ=
.batch_normalization_50_readvariableop_resource:	ђ?
0batch_normalization_50_readvariableop_1_resource:	ђN
?batch_normalization_50_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_50/FusedBatchNormV3/ReadVariableOpб8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_50/ReadVariableOpб'batch_normalization_50/ReadVariableOp_1б conv2d_62/BiasAdd/ReadVariableOpбconv2d_62/Conv2D/ReadVariableOpњ
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0к
conv2d_62/Conv2DConv2Dinput_tensor'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ*
paddingSAME*
strides
Є
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0«
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђ
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђЉ
%batch_normalization_50/ReadVariableOpReadVariableOp.batch_normalization_50_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_50/ReadVariableOp_1ReadVariableOp0batch_normalization_50_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_50/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_50_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_50_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0о
'batch_normalization_50/FusedBatchNormV3FusedBatchNormV3conv2d_62/Relu:activations:0-batch_normalization_50/ReadVariableOp:value:0/batch_normalization_50/ReadVariableOp_1:value:0>batch_normalization_50/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ћ
IdentityIdentity+batch_normalization_50/FusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђЛ
NoOpNoOp7^batch_normalization_50/FusedBatchNormV3/ReadVariableOp9^batch_normalization_50/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_50/ReadVariableOp(^batch_normalization_50/ReadVariableOp_1!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::,                           ђ: : : : : : 2t
8batch_normalization_50/FusedBatchNormV3/ReadVariableOp_18batch_normalization_50/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_50/FusedBatchNormV3/ReadVariableOp6batch_normalization_50/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_50/ReadVariableOp_1'batch_normalization_50/ReadVariableOp_12N
%batch_normalization_50/ReadVariableOp%batch_normalization_50/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp:p l
B
_output_shapes0
.:,                           ђ
&
_user_specified_nameinput_tensor"з
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
StatefulPartitionedCall:0         ђђtensorflow/serving/predict:Вэ
└
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
о
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
д
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
╩
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
п
/trace_0
0trace_1
1trace_2
2trace_32ь
)__inference_decoder_layer_call_fn_1937352
)__inference_decoder_layer_call_fn_1937468
)__inference_decoder_layer_call_fn_1937817
)__inference_decoder_layer_call_fn_1937870Й
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
 z/trace_0z0trace_1z1trace_2z2trace_3
─
3trace_0
4trace_1
5trace_2
6trace_32┘
D__inference_decoder_layer_call_and_return_conditional_losses_1937097
D__inference_decoder_layer_call_and_return_conditional_losses_1937235
D__inference_decoder_layer_call_and_return_conditional_losses_1937972
D__inference_decoder_layer_call_and_return_conditional_losses_1938074Й
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
 z3trace_0z4trace_1z5trace_2z6trace_3
═B╩
"__inference__wrapped_model_1936670input_1"ў
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
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Ц
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
и
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses
Iconv
Jbn"
_tf_keras_layer
Ц
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
и
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
Wconv
Xbn"
_tf_keras_layer
Ц
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
и
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
econv
fbn"
_tf_keras_layer
П
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
П
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
,:*ђђђ2decoder/dense_22/kernel
%:#ђђ2decoder/dense_22/bias
E:Cђђ2)decoder/decoder_block_18/conv2d_62/kernel
6:4ђ2'decoder/decoder_block_18/conv2d_62/bias
D:Bђ25decoder/decoder_block_18/batch_normalization_50/gamma
C:Aђ24decoder/decoder_block_18/batch_normalization_50/beta
L:Jђ (2;decoder/decoder_block_18/batch_normalization_50/moving_mean
P:Nђ (2?decoder/decoder_block_18/batch_normalization_50/moving_variance
E:Cђђ2)decoder/decoder_block_19/conv2d_63/kernel
6:4ђ2'decoder/decoder_block_19/conv2d_63/bias
D:Bђ25decoder/decoder_block_19/batch_normalization_51/gamma
C:Aђ24decoder/decoder_block_19/batch_normalization_51/beta
L:Jђ (2;decoder/decoder_block_19/batch_normalization_51/moving_mean
P:Nђ (2?decoder/decoder_block_19/batch_normalization_51/moving_variance
D:Bђ@2)decoder/decoder_block_20/conv2d_64/kernel
5:3@2'decoder/decoder_block_20/conv2d_64/bias
C:A@25decoder/decoder_block_20/batch_normalization_52/gamma
B:@@24decoder/decoder_block_20/batch_normalization_52/beta
K:I@ (2;decoder/decoder_block_20/batch_normalization_52/moving_mean
O:M@ (2?decoder/decoder_block_20/batch_normalization_52/moving_variance
2:0@ 2decoder/conv2d_65/kernel
$:" 2decoder/conv2d_65/bias
2:0 2decoder/conv2d_66/kernel
$:"2decoder/conv2d_66/bias
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
ЩBэ
)__inference_decoder_layer_call_fn_1937352input_1"Й
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
)__inference_decoder_layer_call_fn_1937468input_1"Й
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
)__inference_decoder_layer_call_fn_1937817embedding_input"Й
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
)__inference_decoder_layer_call_fn_1937870embedding_input"Й
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
D__inference_decoder_layer_call_and_return_conditional_losses_1937097input_1"Й
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
D__inference_decoder_layer_call_and_return_conditional_losses_1937235input_1"Й
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
D__inference_decoder_layer_call_and_return_conditional_losses_1937972embedding_input"Й
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
D__inference_decoder_layer_call_and_return_conditional_losses_1938074embedding_input"Й
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
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
С
{trace_02К
*__inference_dense_22_layer_call_fn_1938083ў
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
 z{trace_0
 
|trace_02Р
E__inference_dense_22_layer_call_and_return_conditional_losses_1938094ў
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
 z|trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
}non_trainable_variables

~layers
metrics
 ђlayer_regularization_losses
Ђlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
Ь
ѓtrace_02¤
2__inference_up_sampling2d_18_layer_call_fn_1938099ў
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
 zѓtrace_0
Ѕ
Ѓtrace_02Ж
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1938111ў
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
 zЃtrace_0
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
▓
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Н
Ѕtrace_0
іtrace_12џ
2__inference_decoder_block_18_layer_call_fn_1938128
2__inference_decoder_block_18_layer_call_fn_1938145»
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
 zЅtrace_0zіtrace_1
І
Іtrace_0
їtrace_12л
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938170
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938195»
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
 zІtrace_0zїtrace_1
С
Ї	variables
јtrainable_variables
Јregularization_losses
љ	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses

kernel
bias
!Њ_jit_compiled_convolution_op"
_tf_keras_layer
ы
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
	џaxis
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
▓
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ь
аtrace_02¤
2__inference_up_sampling2d_19_layer_call_fn_1938200ў
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
 zаtrace_0
Ѕ
Аtrace_02Ж
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1938212ў
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
 zАtrace_0
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
▓
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
Н
Дtrace_0
еtrace_12џ
2__inference_decoder_block_19_layer_call_fn_1938229
2__inference_decoder_block_19_layer_call_fn_1938246»
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
 zДtrace_0zеtrace_1
І
Еtrace_0
фtrace_12л
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938271
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938296»
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
 zЕtrace_0zфtrace_1
С
Ф	variables
гtrainable_variables
Гregularization_losses
«	keras_api
»__call__
+░&call_and_return_all_conditional_losses

kernel
bias
!▒_jit_compiled_convolution_op"
_tf_keras_layer
ы
▓	variables
│trainable_variables
┤regularization_losses
х	keras_api
Х__call__
+и&call_and_return_all_conditional_losses
	Иaxis
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
▓
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
йlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Ь
Йtrace_02¤
2__inference_up_sampling2d_20_layer_call_fn_1938301ў
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
 zЙtrace_0
Ѕ
┐trace_02Ж
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1938313ў
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
 z┐trace_0
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
▓
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
Н
┼trace_0
кtrace_12џ
2__inference_decoder_block_20_layer_call_fn_1938330
2__inference_decoder_block_20_layer_call_fn_1938347»
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
 z┼trace_0zкtrace_1
І
Кtrace_0
╚trace_12л
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938372
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938397»
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
 zКtrace_0z╚trace_1
С
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses

 kernel
!bias
!¤_jit_compiled_convolution_op"
_tf_keras_layer
ы
л	variables
Лtrainable_variables
мregularization_losses
М	keras_api
н__call__
+Н&call_and_return_all_conditional_losses
	оaxis
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
▓
Оnon_trainable_variables
пlayers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
у
▄trace_02╚
+__inference_conv2d_65_layer_call_fn_1938406ў
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
 z▄trace_0
ѓ
Пtrace_02с
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1938417ў
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
 zПtrace_0
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
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
яnon_trainable_variables
▀layers
Яmetrics
 рlayer_regularization_losses
Рlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
у
сtrace_02╚
+__inference_conv2d_66_layer_call_fn_1938426ў
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
 zсtrace_0
ѓ
Сtrace_02с
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1938437ў
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
 zСtrace_0
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
%__inference_signature_wrapper_1937764input_1"ћ
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
нBЛ
*__inference_dense_22_layer_call_fn_1938083inputs"ў
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
E__inference_dense_22_layer_call_and_return_conditional_losses_1938094inputs"ў
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
▄B┘
2__inference_up_sampling2d_18_layer_call_fn_1938099inputs"ў
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
эBЗ
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1938111inputs"ў
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
щBШ
2__inference_decoder_block_18_layer_call_fn_1938128input_tensor"»
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
щBШ
2__inference_decoder_block_18_layer_call_fn_1938145input_tensor"»
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
ћBЉ
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938170input_tensor"»
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
ћBЉ
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938195input_tensor"»
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
Ї	variables
јtrainable_variables
Јregularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
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
И
Жnon_trainable_variables
вlayers
Вmetrics
 ьlayer_regularization_losses
Ьlayer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
у
№trace_0
­trace_12г
8__inference_batch_normalization_50_layer_call_fn_1938450
8__inference_batch_normalization_50_layer_call_fn_1938463х
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
 z№trace_0z­trace_1
Ю
ыtrace_0
Ыtrace_12Р
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938481
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938499х
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
 zыtrace_0zЫtrace_1
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
▄B┘
2__inference_up_sampling2d_19_layer_call_fn_1938200inputs"ў
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
эBЗ
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1938212inputs"ў
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
щBШ
2__inference_decoder_block_19_layer_call_fn_1938229input_tensor"»
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
щBШ
2__inference_decoder_block_19_layer_call_fn_1938246input_tensor"»
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
ћBЉ
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938271input_tensor"»
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
ћBЉ
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938296input_tensor"»
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
зnon_trainable_variables
Зlayers
шmetrics
 Шlayer_regularization_losses
эlayer_metrics
Ф	variables
гtrainable_variables
Гregularization_losses
»__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
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
И
Эnon_trainable_variables
щlayers
Щmetrics
 чlayer_regularization_losses
Чlayer_metrics
▓	variables
│trainable_variables
┤regularization_losses
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
у
§trace_0
■trace_12г
8__inference_batch_normalization_51_layer_call_fn_1938512
8__inference_batch_normalization_51_layer_call_fn_1938525х
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
 z§trace_0z■trace_1
Ю
 trace_0
ђtrace_12Р
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938543
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938561х
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
 z trace_0zђtrace_1
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
▄B┘
2__inference_up_sampling2d_20_layer_call_fn_1938301inputs"ў
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
эBЗ
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1938313inputs"ў
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
щBШ
2__inference_decoder_block_20_layer_call_fn_1938330input_tensor"»
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
щBШ
2__inference_decoder_block_20_layer_call_fn_1938347input_tensor"»
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
ћBЉ
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938372input_tensor"»
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
ћBЉ
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938397input_tensor"»
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
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ђnon_trainable_variables
ѓlayers
Ѓmetrics
 ёlayer_regularization_losses
Ёlayer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
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
И
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
л	variables
Лtrainable_variables
мregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
у
Іtrace_0
їtrace_12г
8__inference_batch_normalization_52_layer_call_fn_1938574
8__inference_batch_normalization_52_layer_call_fn_1938587х
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
Ю
Їtrace_0
јtrace_12Р
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938605
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938623х
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
НBм
+__inference_conv2d_65_layer_call_fn_1938406inputs"ў
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
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1938417inputs"ў
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
+__inference_conv2d_66_layer_call_fn_1938426inputs"ў
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
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1938437inputs"ў
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
 BЧ
8__inference_batch_normalization_50_layer_call_fn_1938450inputs"х
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
 BЧ
8__inference_batch_normalization_50_layer_call_fn_1938463inputs"х
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
џBЌ
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938481inputs"х
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
џBЌ
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938499inputs"х
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
 BЧ
8__inference_batch_normalization_51_layer_call_fn_1938512inputs"х
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
 BЧ
8__inference_batch_normalization_51_layer_call_fn_1938525inputs"х
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
џBЌ
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938543inputs"х
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
џBЌ
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938561inputs"х
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
 BЧ
8__inference_batch_normalization_52_layer_call_fn_1938574inputs"х
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
 BЧ
8__inference_batch_normalization_52_layer_call_fn_1938587inputs"х
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
џBЌ
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938605inputs"х
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
џBЌ
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938623inputs"х
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
"__inference__wrapped_model_1936670ї !"#$%&'()1б.
'б$
"і
input_1         ђ
ф "=ф:
8
output_1,і)
output_1         ђђч
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938481БRбO
HбE
;і8
inputs,                           ђ
p

 
ф "GбD
=і:
tensor_0,                           ђ
џ ч
S__inference_batch_normalization_50_layer_call_and_return_conditional_losses_1938499БRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "GбD
=і:
tensor_0,                           ђ
џ Н
8__inference_batch_normalization_50_layer_call_fn_1938450ўRбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђН
8__inference_batch_normalization_50_layer_call_fn_1938463ўRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђч
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938543БRбO
HбE
;і8
inputs,                           ђ
p

 
ф "GбD
=і:
tensor_0,                           ђ
џ ч
S__inference_batch_normalization_51_layer_call_and_return_conditional_losses_1938561БRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "GбD
=і:
tensor_0,                           ђ
џ Н
8__inference_batch_normalization_51_layer_call_fn_1938512ўRбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђН
8__inference_batch_normalization_51_layer_call_fn_1938525ўRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђщ
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938605А"#$%QбN
GбD
:і7
inputs+                           @
p

 
ф "FбC
<і9
tensor_0+                           @
џ щ
S__inference_batch_normalization_52_layer_call_and_return_conditional_losses_1938623А"#$%QбN
GбD
:і7
inputs+                           @
p 

 
ф "FбC
<і9
tensor_0+                           @
џ М
8__inference_batch_normalization_52_layer_call_fn_1938574ќ"#$%QбN
GбD
:і7
inputs+                           @
p

 
ф ";і8
unknown+                           @М
8__inference_batch_normalization_52_layer_call_fn_1938587ќ"#$%QбN
GбD
:і7
inputs+                           @
p 

 
ф ";і8
unknown+                           @Р
F__inference_conv2d_65_layer_call_and_return_conditional_losses_1938417Ќ&'IбF
?б<
:і7
inputs+                           @
ф "FбC
<і9
tensor_0+                            
џ ╝
+__inference_conv2d_65_layer_call_fn_1938406ї&'IбF
?б<
:і7
inputs+                           @
ф ";і8
unknown+                            Р
F__inference_conv2d_66_layer_call_and_return_conditional_losses_1938437Ќ()IбF
?б<
:і7
inputs+                            
ф "FбC
<і9
tensor_0+                           
џ ╝
+__inference_conv2d_66_layer_call_fn_1938426ї()IбF
?б<
:і7
inputs+                            
ф ";і8
unknown+                           щ
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938170ДTбQ
JбG
Aі>
input_tensor,                           ђ
p
ф "GбD
=і:
tensor_0,                           ђ
џ щ
M__inference_decoder_block_18_layer_call_and_return_conditional_losses_1938195ДTбQ
JбG
Aі>
input_tensor,                           ђ
p 
ф "GбD
=і:
tensor_0,                           ђ
џ М
2__inference_decoder_block_18_layer_call_fn_1938128юTбQ
JбG
Aі>
input_tensor,                           ђ
p
ф "<і9
unknown,                           ђМ
2__inference_decoder_block_18_layer_call_fn_1938145юTбQ
JбG
Aі>
input_tensor,                           ђ
p 
ф "<і9
unknown,                           ђщ
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938271ДTбQ
JбG
Aі>
input_tensor,                           ђ
p
ф "GбD
=і:
tensor_0,                           ђ
џ щ
M__inference_decoder_block_19_layer_call_and_return_conditional_losses_1938296ДTбQ
JбG
Aі>
input_tensor,                           ђ
p 
ф "GбD
=і:
tensor_0,                           ђ
џ М
2__inference_decoder_block_19_layer_call_fn_1938229юTбQ
JбG
Aі>
input_tensor,                           ђ
p
ф "<і9
unknown,                           ђМ
2__inference_decoder_block_19_layer_call_fn_1938246юTбQ
JбG
Aі>
input_tensor,                           ђ
p 
ф "<і9
unknown,                           ђЭ
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938372д !"#$%TбQ
JбG
Aі>
input_tensor,                           ђ
p
ф "FбC
<і9
tensor_0+                           @
џ Э
M__inference_decoder_block_20_layer_call_and_return_conditional_losses_1938397д !"#$%TбQ
JбG
Aі>
input_tensor,                           ђ
p 
ф "FбC
<і9
tensor_0+                           @
џ м
2__inference_decoder_block_20_layer_call_fn_1938330Џ !"#$%TбQ
JбG
Aі>
input_tensor,                           ђ
p
ф ";і8
unknown+                           @м
2__inference_decoder_block_20_layer_call_fn_1938347Џ !"#$%TбQ
JбG
Aі>
input_tensor,                           ђ
p 
ф ";і8
unknown+                           @Ь
D__inference_decoder_layer_call_and_return_conditional_losses_1937097Ц !"#$%&'()Aб>
'б$
"і
input_1         ђ
ф

trainingp"FбC
<і9
tensor_0+                           
џ Ь
D__inference_decoder_layer_call_and_return_conditional_losses_1937235Ц !"#$%&'()Aб>
'б$
"і
input_1         ђ
ф

trainingp "FбC
<і9
tensor_0+                           
џ Т
D__inference_decoder_layer_call_and_return_conditional_losses_1937972Ю !"#$%&'()IбF
/б,
*і'
embedding_input         ђ
ф

trainingp"6б3
,і)
tensor_0         ђђ
џ Т
D__inference_decoder_layer_call_and_return_conditional_losses_1938074Ю !"#$%&'()IбF
/б,
*і'
embedding_input         ђ
ф

trainingp "6б3
,і)
tensor_0         ђђ
џ ╚
)__inference_decoder_layer_call_fn_1937352џ !"#$%&'()Aб>
'б$
"і
input_1         ђ
ф

trainingp";і8
unknown+                           ╚
)__inference_decoder_layer_call_fn_1937468џ !"#$%&'()Aб>
'б$
"і
input_1         ђ
ф

trainingp ";і8
unknown+                           л
)__inference_decoder_layer_call_fn_1937817б !"#$%&'()IбF
/б,
*і'
embedding_input         ђ
ф

trainingp";і8
unknown+                           л
)__inference_decoder_layer_call_fn_1937870б !"#$%&'()IбF
/б,
*і'
embedding_input         ђ
ф

trainingp ";і8
unknown+                           »
E__inference_dense_22_layer_call_and_return_conditional_losses_1938094f0б-
&б#
!і
inputs         ђ
ф ".б+
$і!
tensor_0         ђђ
џ Ѕ
*__inference_dense_22_layer_call_fn_1938083[0б-
&б#
!і
inputs         ђ
ф "#і 
unknown         ђђ┴
%__inference_signature_wrapper_1937764Ќ !"#$%&'()<б9
б 
2ф/
-
input_1"і
input_1         ђ"=ф:
8
output_1,і)
output_1         ђђэ
M__inference_up_sampling2d_18_layer_call_and_return_conditional_losses_1938111ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_up_sampling2d_18_layer_call_fn_1938099џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    э
M__inference_up_sampling2d_19_layer_call_and_return_conditional_losses_1938212ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_up_sampling2d_19_layer_call_fn_1938200џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    э
M__inference_up_sampling2d_20_layer_call_and_return_conditional_losses_1938313ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_up_sampling2d_20_layer_call_fn_1938301џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    