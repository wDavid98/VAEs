тк
йї
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
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
,
Exp
x"T
y"T"
Ttype:

2
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
ѓ
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
Ё
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
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
d
Shape

input"T&
output"out_typeіьout_type"	
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758Ѓ▓
Ѓ
encoder/dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameencoder/dense_13/bias
|
)encoder/dense_13/bias/Read/ReadVariableOpReadVariableOpencoder/dense_13/bias*
_output_shapes	
:ђ*
dtype0
ї
encoder/dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*(
shared_nameencoder/dense_13/kernel
Ё
+encoder/dense_13/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_13/kernel* 
_output_shapes
:
ђђ*
dtype0
Ѓ
encoder/dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameencoder/dense_12/bias
|
)encoder/dense_12/bias/Read/ReadVariableOpReadVariableOpencoder/dense_12/bias*
_output_shapes	
:ђ*
dtype0
ї
encoder/dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*(
shared_nameencoder/dense_12/kernel
Ё
+encoder/dense_12/kernel/Read/ReadVariableOpReadVariableOpencoder/dense_12/kernel* 
_output_shapes
:
ђђ*
dtype0
ё
encoder/conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameencoder/conv2d_49/bias
}
*encoder/conv2d_49/bias/Read/ReadVariableOpReadVariableOpencoder/conv2d_49/bias*
_output_shapes
:*
dtype0
Ћ
encoder/conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*)
shared_nameencoder/conv2d_49/kernel
ј
,encoder/conv2d_49/kernel/Read/ReadVariableOpReadVariableOpencoder/conv2d_49/kernel*'
_output_shapes
:ђ*
dtype0
О
?encoder/encoder_block_23/batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*P
shared_nameA?encoder/encoder_block_23/batch_normalization_35/moving_variance
л
Sencoder/encoder_block_23/batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_23/batch_normalization_35/moving_variance*
_output_shapes	
:ђ*
dtype0
¤
;encoder/encoder_block_23/batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;encoder/encoder_block_23/batch_normalization_35/moving_mean
╚
Oencoder/encoder_block_23/batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_23/batch_normalization_35/moving_mean*
_output_shapes	
:ђ*
dtype0
┴
4encoder/encoder_block_23/batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*E
shared_name64encoder/encoder_block_23/batch_normalization_35/beta
║
Hencoder/encoder_block_23/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_23/batch_normalization_35/beta*
_output_shapes	
:ђ*
dtype0
├
5encoder/encoder_block_23/batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*F
shared_name75encoder/encoder_block_23/batch_normalization_35/gamma
╝
Iencoder/encoder_block_23/batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_23/batch_normalization_35/gamma*
_output_shapes	
:ђ*
dtype0
Д
'encoder/encoder_block_23/conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'encoder/encoder_block_23/conv2d_47/bias
а
;encoder/encoder_block_23/conv2d_47/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_23/conv2d_47/bias*
_output_shapes	
:ђ*
dtype0
И
)encoder/encoder_block_23/conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*:
shared_name+)encoder/encoder_block_23/conv2d_47/kernel
▒
=encoder/encoder_block_23/conv2d_47/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_23/conv2d_47/kernel*(
_output_shapes
:ђђ*
dtype0
О
?encoder/encoder_block_22/batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*P
shared_nameA?encoder/encoder_block_22/batch_normalization_34/moving_variance
л
Sencoder/encoder_block_22/batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_22/batch_normalization_34/moving_variance*
_output_shapes	
:ђ*
dtype0
¤
;encoder/encoder_block_22/batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;encoder/encoder_block_22/batch_normalization_34/moving_mean
╚
Oencoder/encoder_block_22/batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_22/batch_normalization_34/moving_mean*
_output_shapes	
:ђ*
dtype0
┴
4encoder/encoder_block_22/batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*E
shared_name64encoder/encoder_block_22/batch_normalization_34/beta
║
Hencoder/encoder_block_22/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_22/batch_normalization_34/beta*
_output_shapes	
:ђ*
dtype0
├
5encoder/encoder_block_22/batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*F
shared_name75encoder/encoder_block_22/batch_normalization_34/gamma
╝
Iencoder/encoder_block_22/batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_22/batch_normalization_34/gamma*
_output_shapes	
:ђ*
dtype0
Д
'encoder/encoder_block_22/conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'encoder/encoder_block_22/conv2d_46/bias
а
;encoder/encoder_block_22/conv2d_46/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_22/conv2d_46/bias*
_output_shapes	
:ђ*
dtype0
И
)encoder/encoder_block_22/conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*:
shared_name+)encoder/encoder_block_22/conv2d_46/kernel
▒
=encoder/encoder_block_22/conv2d_46/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_22/conv2d_46/kernel*(
_output_shapes
:ђђ*
dtype0
О
?encoder/encoder_block_21/batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*P
shared_nameA?encoder/encoder_block_21/batch_normalization_33/moving_variance
л
Sencoder/encoder_block_21/batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_21/batch_normalization_33/moving_variance*
_output_shapes	
:ђ*
dtype0
¤
;encoder/encoder_block_21/batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;encoder/encoder_block_21/batch_normalization_33/moving_mean
╚
Oencoder/encoder_block_21/batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_21/batch_normalization_33/moving_mean*
_output_shapes	
:ђ*
dtype0
┴
4encoder/encoder_block_21/batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*E
shared_name64encoder/encoder_block_21/batch_normalization_33/beta
║
Hencoder/encoder_block_21/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_21/batch_normalization_33/beta*
_output_shapes	
:ђ*
dtype0
├
5encoder/encoder_block_21/batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*F
shared_name75encoder/encoder_block_21/batch_normalization_33/gamma
╝
Iencoder/encoder_block_21/batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_21/batch_normalization_33/gamma*
_output_shapes	
:ђ*
dtype0
Д
'encoder/encoder_block_21/conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'encoder/encoder_block_21/conv2d_45/bias
а
;encoder/encoder_block_21/conv2d_45/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_21/conv2d_45/bias*
_output_shapes	
:ђ*
dtype0
И
)encoder/encoder_block_21/conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*:
shared_name+)encoder/encoder_block_21/conv2d_45/kernel
▒
=encoder/encoder_block_21/conv2d_45/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_21/conv2d_45/kernel*(
_output_shapes
:ђђ*
dtype0
О
?encoder/encoder_block_20/batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*P
shared_nameA?encoder/encoder_block_20/batch_normalization_32/moving_variance
л
Sencoder/encoder_block_20/batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp?encoder/encoder_block_20/batch_normalization_32/moving_variance*
_output_shapes	
:ђ*
dtype0
¤
;encoder/encoder_block_20/batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*L
shared_name=;encoder/encoder_block_20/batch_normalization_32/moving_mean
╚
Oencoder/encoder_block_20/batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp;encoder/encoder_block_20/batch_normalization_32/moving_mean*
_output_shapes	
:ђ*
dtype0
┴
4encoder/encoder_block_20/batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*E
shared_name64encoder/encoder_block_20/batch_normalization_32/beta
║
Hencoder/encoder_block_20/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOp4encoder/encoder_block_20/batch_normalization_32/beta*
_output_shapes	
:ђ*
dtype0
├
5encoder/encoder_block_20/batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*F
shared_name75encoder/encoder_block_20/batch_normalization_32/gamma
╝
Iencoder/encoder_block_20/batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOp5encoder/encoder_block_20/batch_normalization_32/gamma*
_output_shapes	
:ђ*
dtype0
Д
'encoder/encoder_block_20/conv2d_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'encoder/encoder_block_20/conv2d_44/bias
а
;encoder/encoder_block_20/conv2d_44/bias/Read/ReadVariableOpReadVariableOp'encoder/encoder_block_20/conv2d_44/bias*
_output_shapes	
:ђ*
dtype0
и
)encoder/encoder_block_20/conv2d_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*:
shared_name+)encoder/encoder_block_20/conv2d_44/kernel
░
=encoder/encoder_block_20/conv2d_44/kernel/Read/ReadVariableOpReadVariableOp)encoder/encoder_block_20/conv2d_44/kernel*'
_output_shapes
:ђ*
dtype0
ј
serving_default_input_1Placeholder*1
_output_shapes
:         ђђ*
dtype0*&
shape:         ђђ
Г
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1)encoder/encoder_block_20/conv2d_44/kernel'encoder/encoder_block_20/conv2d_44/bias5encoder/encoder_block_20/batch_normalization_32/gamma4encoder/encoder_block_20/batch_normalization_32/beta;encoder/encoder_block_20/batch_normalization_32/moving_mean?encoder/encoder_block_20/batch_normalization_32/moving_variance)encoder/encoder_block_21/conv2d_45/kernel'encoder/encoder_block_21/conv2d_45/bias5encoder/encoder_block_21/batch_normalization_33/gamma4encoder/encoder_block_21/batch_normalization_33/beta;encoder/encoder_block_21/batch_normalization_33/moving_mean?encoder/encoder_block_21/batch_normalization_33/moving_variance)encoder/encoder_block_22/conv2d_46/kernel'encoder/encoder_block_22/conv2d_46/bias5encoder/encoder_block_22/batch_normalization_34/gamma4encoder/encoder_block_22/batch_normalization_34/beta;encoder/encoder_block_22/batch_normalization_34/moving_mean?encoder/encoder_block_22/batch_normalization_34/moving_variance)encoder/encoder_block_23/conv2d_47/kernel'encoder/encoder_block_23/conv2d_47/bias5encoder/encoder_block_23/batch_normalization_35/gamma4encoder/encoder_block_23/batch_normalization_35/beta;encoder/encoder_block_23/batch_normalization_35/moving_mean?encoder/encoder_block_23/batch_normalization_35/moving_varianceencoder/conv2d_49/kernelencoder/conv2d_49/biasencoder/dense_12/kernelencoder/dense_12/biasencoder/dense_13/kernelencoder/dense_13/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *.
f)R'
%__inference_signature_wrapper_1911876

NoOpNoOp
┤Ѕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ьѕ
valueсѕB▀ѕ BОѕ
┴
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

block5
	final_cov

flattening

z_mean
z_logvar
	embedding

signatures*
Ж
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
/28
029*
ф
0
1
2
3
4
5
6
7
8
 9
!10
"11
%12
&13
'14
(15
+16
,17
-18
.19
/20
021*
* 
░
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
6trace_0
7trace_1
8trace_2
9trace_3* 
6
:trace_0
;trace_1
<trace_2
=trace_3* 
* 
»
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Dconv
Epooling
Fbn*
»
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Mconv
Npooling
Obn*
»
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
Vconv
Wpooling
Xbn*
»
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_conv
`pooling
abn*
0
b	keras_api
cconv
dpooling
ebn* 
╚
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

+kernel
,bias
 l_jit_compiled_convolution_op*
ј
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
д
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

-kernel
.bias*
д
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

/kernel
0bias*
Њ
	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses* 

Ёserving_default* 
ic
VARIABLE_VALUE)encoder/encoder_block_20/conv2d_44/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'encoder/encoder_block_20/conv2d_44/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoder/encoder_block_20/batch_normalization_32/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4encoder/encoder_block_20/batch_normalization_32/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE;encoder/encoder_block_20/batch_normalization_32/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?encoder/encoder_block_20/batch_normalization_32/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)encoder/encoder_block_21/conv2d_45/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'encoder/encoder_block_21/conv2d_45/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE5encoder/encoder_block_21/batch_normalization_33/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4encoder/encoder_block_21/batch_normalization_33/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_21/batch_normalization_33/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE?encoder/encoder_block_21/batch_normalization_33/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)encoder/encoder_block_22/conv2d_46/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'encoder/encoder_block_22/conv2d_46/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5encoder/encoder_block_22/batch_normalization_34/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4encoder/encoder_block_22/batch_normalization_34/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_22/batch_normalization_34/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE?encoder/encoder_block_22/batch_normalization_34/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUE)encoder/encoder_block_23/conv2d_47/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'encoder/encoder_block_23/conv2d_47/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE5encoder/encoder_block_23/batch_normalization_35/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE4encoder/encoder_block_23/batch_normalization_35/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;encoder/encoder_block_23/batch_normalization_35/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE?encoder/encoder_block_23/batch_normalization_35/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEencoder/conv2d_49/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEencoder/conv2d_49/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEencoder/dense_12/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEencoder/dense_12/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEencoder/dense_13/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEencoder/dense_13/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
<
0
1
2
3
#4
$5
)6
*7*
J
0
	1

2
3
4
5
6
7
8
9*
* 
* 
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
0
1
2
3
4
5*
 
0
1
2
3*
* 
ў
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

Іtrace_0
їtrace_1* 

Їtrace_0
јtrace_1* 
¤
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses

kernel
bias
!Ћ_jit_compiled_convolution_op*
ћ
ќ	variables
Ќtrainable_variables
ўregularization_losses
Ў	keras_api
џ__call__
+Џ&call_and_return_all_conditional_losses* 
▄
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses
	бaxis
	gamma
beta
moving_mean
moving_variance*
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
ў
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

еtrace_0
Еtrace_1* 

фtrace_0
Фtrace_1* 
¤
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses

kernel
bias
!▓_jit_compiled_convolution_op*
ћ
│	variables
┤trainable_variables
хregularization_losses
Х	keras_api
и__call__
+И&call_and_return_all_conditional_losses* 
▄
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
й__call__
+Й&call_and_return_all_conditional_losses
	┐axis
	gamma
beta
moving_mean
moving_variance*
.
0
 1
!2
"3
#4
$5*
 
0
 1
!2
"3*
* 
ў
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

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

kernel
 bias
!¤_jit_compiled_convolution_op*
ћ
л	variables
Лtrainable_variables
мregularization_losses
М	keras_api
н__call__
+Н&call_and_return_all_conditional_losses* 
▄
о	variables
Оtrainable_variables
пregularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses
	▄axis
	!gamma
"beta
#moving_mean
$moving_variance*
.
%0
&1
'2
(3
)4
*5*
 
%0
&1
'2
(3*
* 
ў
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses*

Рtrace_0
сtrace_1* 

Сtrace_0
тtrace_1* 
¤
Т	variables
уtrainable_variables
Уregularization_losses
ж	keras_api
Ж__call__
+в&call_and_return_all_conditional_losses

%kernel
&bias
!В_jit_compiled_convolution_op*
ћ
ь	variables
Ьtrainable_variables
№regularization_losses
­	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses* 
▄
з	variables
Зtrainable_variables
шregularization_losses
Ш	keras_api
э__call__
+Э&call_and_return_all_conditional_losses
	щaxis
	'gamma
(beta
)moving_mean
*moving_variance*
* 
5
Щ	keras_api
!ч_jit_compiled_convolution_op* 

Ч	keras_api* 

§	keras_api* 

+0
,1*

+0
,1*
* 
ў
■non_trainable_variables
 layers
ђmetrics
 Ђlayer_regularization_losses
ѓlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

Ѓtrace_0* 

ёtrace_0* 
* 
* 
* 
* 
ќ
Ёnon_trainable_variables
єlayers
Єmetrics
 ѕlayer_regularization_losses
Ѕlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

іtrace_0* 

Іtrace_0* 

-0
.1*

-0
.1*
* 
ў
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

Љtrace_0* 

њtrace_0* 

/0
01*

/0
01*
* 
ў
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses*

ўtrace_0* 

Ўtrace_0* 
* 
* 
* 
Џ
џnon_trainable_variables
Џlayers
юmetrics
 Юlayer_regularization_losses
ъlayer_metrics
	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses* 

Ъtrace_0* 

аtrace_0* 
* 

0
1*

D0
E1
F2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
ъ
Аnon_trainable_variables
бlayers
Бmetrics
 цlayer_regularization_losses
Цlayer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
ќ	variables
Ќtrainable_variables
ўregularization_losses
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses* 

Фtrace_0* 

гtrace_0* 
 
0
1
2
3*

0
1*
* 
ъ
Гnon_trainable_variables
«layers
»metrics
 ░layer_regularization_losses
▒layer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

▓trace_0
│trace_1* 

┤trace_0
хtrace_1* 
* 

0
1*

M0
N1
O2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
ъ
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
╗non_trainable_variables
╝layers
йmetrics
 Йlayer_regularization_losses
┐layer_metrics
│	variables
┤trainable_variables
хregularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses* 

└trace_0* 

┴trace_0* 
 
0
1
2
3*

0
1*
* 
ъ
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
кlayer_metrics
╣	variables
║trainable_variables
╗regularization_losses
й__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses*

Кtrace_0
╚trace_1* 

╔trace_0
╩trace_1* 
* 

#0
$1*

V0
W1
X2*
* 
* 
* 
* 
* 
* 
* 

0
 1*

0
 1*
* 
ъ
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
лnon_trainable_variables
Лlayers
мmetrics
 Мlayer_regularization_losses
нlayer_metrics
л	variables
Лtrainable_variables
мregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses* 

Нtrace_0* 

оtrace_0* 
 
!0
"1
#2
$3*

!0
"1*
* 
ъ
Оnon_trainable_variables
пlayers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
о	variables
Оtrainable_variables
пregularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses*

▄trace_0
Пtrace_1* 

яtrace_0
▀trace_1* 
* 

)0
*1*

_0
`1
a2*
* 
* 
* 
* 
* 
* 
* 

%0
&1*

%0
&1*
* 
ъ
Яnon_trainable_variables
рlayers
Рmetrics
 сlayer_regularization_losses
Сlayer_metrics
Т	variables
уtrainable_variables
Уregularization_losses
Ж__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
ю
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
ь	variables
Ьtrainable_variables
№regularization_losses
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 

Жtrace_0* 

вtrace_0* 
 
'0
(1
)2
*3*

'0
(1*
* 
ъ
Вnon_trainable_variables
ьlayers
Ьmetrics
 №layer_regularization_losses
­layer_metrics
з	variables
Зtrainable_variables
шregularization_losses
э__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses*

ыtrace_0
Ыtrace_1* 

зtrace_0
Зtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
#0
$1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
)0
*1*
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
═
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)encoder/encoder_block_20/conv2d_44/kernel'encoder/encoder_block_20/conv2d_44/bias5encoder/encoder_block_20/batch_normalization_32/gamma4encoder/encoder_block_20/batch_normalization_32/beta;encoder/encoder_block_20/batch_normalization_32/moving_mean?encoder/encoder_block_20/batch_normalization_32/moving_variance)encoder/encoder_block_21/conv2d_45/kernel'encoder/encoder_block_21/conv2d_45/bias5encoder/encoder_block_21/batch_normalization_33/gamma4encoder/encoder_block_21/batch_normalization_33/beta;encoder/encoder_block_21/batch_normalization_33/moving_mean?encoder/encoder_block_21/batch_normalization_33/moving_variance)encoder/encoder_block_22/conv2d_46/kernel'encoder/encoder_block_22/conv2d_46/bias5encoder/encoder_block_22/batch_normalization_34/gamma4encoder/encoder_block_22/batch_normalization_34/beta;encoder/encoder_block_22/batch_normalization_34/moving_mean?encoder/encoder_block_22/batch_normalization_34/moving_variance)encoder/encoder_block_23/conv2d_47/kernel'encoder/encoder_block_23/conv2d_47/bias5encoder/encoder_block_23/batch_normalization_35/gamma4encoder/encoder_block_23/batch_normalization_35/beta;encoder/encoder_block_23/batch_normalization_35/moving_mean?encoder/encoder_block_23/batch_normalization_35/moving_varianceencoder/conv2d_49/kernelencoder/conv2d_49/biasencoder/dense_12/kernelencoder/dense_12/biasencoder/dense_13/kernelencoder/dense_13/biasConst*+
Tin$
"2 *
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
 __inference__traced_save_1913224
╚
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename)encoder/encoder_block_20/conv2d_44/kernel'encoder/encoder_block_20/conv2d_44/bias5encoder/encoder_block_20/batch_normalization_32/gamma4encoder/encoder_block_20/batch_normalization_32/beta;encoder/encoder_block_20/batch_normalization_32/moving_mean?encoder/encoder_block_20/batch_normalization_32/moving_variance)encoder/encoder_block_21/conv2d_45/kernel'encoder/encoder_block_21/conv2d_45/bias5encoder/encoder_block_21/batch_normalization_33/gamma4encoder/encoder_block_21/batch_normalization_33/beta;encoder/encoder_block_21/batch_normalization_33/moving_mean?encoder/encoder_block_21/batch_normalization_33/moving_variance)encoder/encoder_block_22/conv2d_46/kernel'encoder/encoder_block_22/conv2d_46/bias5encoder/encoder_block_22/batch_normalization_34/gamma4encoder/encoder_block_22/batch_normalization_34/beta;encoder/encoder_block_22/batch_normalization_34/moving_mean?encoder/encoder_block_22/batch_normalization_34/moving_variance)encoder/encoder_block_23/conv2d_47/kernel'encoder/encoder_block_23/conv2d_47/bias5encoder/encoder_block_23/batch_normalization_35/gamma4encoder/encoder_block_23/batch_normalization_35/beta;encoder/encoder_block_23/batch_normalization_35/moving_mean?encoder/encoder_block_23/batch_normalization_35/moving_varianceencoder/conv2d_49/kernelencoder/conv2d_49/biasencoder/dense_12/kernelencoder/dense_12/biasencoder/dense_13/kernelencoder/dense_13/bias**
Tin#
!2*
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
#__inference__traced_restore_1913324ЫД
б	
О
8__inference_batch_normalization_35_layer_call_fn_1912983

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1910737і
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
И
G
+__inference_flatten_4_layer_call_fn_1912655

inputs
identityи
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_1910951a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
л
џ
*__inference_dense_12_layer_call_fn_1912670

inputs
unknown:
ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_1910963p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
Ъ
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1911122
input_tensorD
(conv2d_46_conv2d_readvariableop_resource:ђђ8
)conv2d_46_biasadd_readvariableop_resource:	ђ=
.batch_normalization_34_readvariableop_resource:	ђ?
0batch_normalization_34_readvariableop_1_resource:	ђN
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_34/FusedBatchNormV3/ReadVariableOpб8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_34/ReadVariableOpб'batch_normalization_34/ReadVariableOp_1б conv2d_46/BiasAdd/ReadVariableOpбconv2d_46/Conv2D/ReadVariableOpњ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_46/Conv2DConv2Dinput_tensor'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Є
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђm
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ»
max_pooling2d_22/MaxPoolMaxPoolconv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_22/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_34/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђЛ
NoOpNoOp7^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           ђ: : : : : : 2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           ђ
&
_user_specified_nameinput_tensor
Ё
u
,__inference_sampling_4_layer_call_fn_1912705
inputs_0
inputs_1
identityѕбStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_sampling_4_layer_call_and_return_conditional_losses_1911011p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:RN
(
_output_shapes
:         ђ
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs_0
Ћ
i
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1912741

inputs
identityб
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
┴
N
2__inference_max_pooling2d_22_layer_call_fn_1912880

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
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1910618Ѓ
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
Ш
Ъ
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912630
input_tensorD
(conv2d_47_conv2d_readvariableop_resource:ђђ8
)conv2d_47_biasadd_readvariableop_resource:	ђ=
.batch_normalization_35_readvariableop_resource:	ђ?
0batch_normalization_35_readvariableop_1_resource:	ђN
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_35/FusedBatchNormV3/ReadVariableOpб8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_35/ReadVariableOpб'batch_normalization_35/ReadVariableOp_1б conv2d_47/BiasAdd/ReadVariableOpбconv2d_47/Conv2D/ReadVariableOpњ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_47/Conv2DConv2Dinput_tensor'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Є
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђm
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ»
max_pooling2d_23/MaxPoolMaxPoolconv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_23/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_35/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђЛ
NoOpNoOp7^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
б	
О
8__inference_batch_normalization_33_layer_call_fn_1912839

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1910585і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
ў
к
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912785

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ш
Ъ
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912458
input_tensorD
(conv2d_45_conv2d_readvariableop_resource:ђђ8
)conv2d_45_biasadd_readvariableop_resource:	ђ=
.batch_normalization_33_readvariableop_resource:	ђ?
0batch_normalization_33_readvariableop_1_resource:	ђN
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_33/FusedBatchNormV3/ReadVariableOpб8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_33/ReadVariableOpб'batch_normalization_33/ReadVariableOp_1б conv2d_45/BiasAdd/ReadVariableOpбconv2d_45/Conv2D/ReadVariableOpњ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_45/Conv2DConv2Dinput_tensor'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
Є
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђm
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђ»
max_pooling2d_21/MaxPoolMaxPoolconv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_21/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_33/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           ђЛ
NoOpNoOp7^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@ђ: : : : : : 2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@ђ
&
_user_specified_nameinput_tensor
б	
О
8__inference_batch_normalization_32_layer_call_fn_1912767

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1910509і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
а	
О
8__inference_batch_normalization_32_layer_call_fn_1912754

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1910491і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
а'
ы
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912518
input_tensorD
(conv2d_46_conv2d_readvariableop_resource:ђђ8
)conv2d_46_biasadd_readvariableop_resource:	ђ=
.batch_normalization_34_readvariableop_resource:	ђ?
0batch_normalization_34_readvariableop_1_resource:	ђN
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_34/AssignNewValueб'batch_normalization_34/AssignNewValue_1б6batch_normalization_34/FusedBatchNormV3/ReadVariableOpб8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_34/ReadVariableOpб'batch_normalization_34/ReadVariableOp_1б conv2d_46/BiasAdd/ReadVariableOpбconv2d_46/Conv2D/ReadVariableOpњ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_46/Conv2DConv2Dinput_tensor'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Є
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђm
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ»
max_pooling2d_22/MaxPoolMaxPoolconv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_22/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_34/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђБ
NoOpNoOp&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           ђ: : : : : : 2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           ђ
&
_user_specified_nameinput_tensor
б	
О
8__inference_batch_normalization_34_layer_call_fn_1912911

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1910661і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╝<
г
D__inference_encoder_layer_call_and_return_conditional_losses_1911273
tensor_input3
encoder_block_20_1911201:ђ'
encoder_block_20_1911203:	ђ'
encoder_block_20_1911205:	ђ'
encoder_block_20_1911207:	ђ'
encoder_block_20_1911209:	ђ'
encoder_block_20_1911211:	ђ4
encoder_block_21_1911214:ђђ'
encoder_block_21_1911216:	ђ'
encoder_block_21_1911218:	ђ'
encoder_block_21_1911220:	ђ'
encoder_block_21_1911222:	ђ'
encoder_block_21_1911224:	ђ4
encoder_block_22_1911227:ђђ'
encoder_block_22_1911229:	ђ'
encoder_block_22_1911231:	ђ'
encoder_block_22_1911233:	ђ'
encoder_block_22_1911235:	ђ'
encoder_block_22_1911237:	ђ4
encoder_block_23_1911240:ђђ'
encoder_block_23_1911242:	ђ'
encoder_block_23_1911244:	ђ'
encoder_block_23_1911246:	ђ'
encoder_block_23_1911248:	ђ'
encoder_block_23_1911250:	ђ,
conv2d_49_1911253:ђ
conv2d_49_1911255:$
dense_12_1911259:
ђђ
dense_12_1911261:	ђ$
dense_13_1911264:
ђђ
dense_13_1911266:	ђ
identity

identity_1

identity_2ѕб!conv2d_49/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб(encoder_block_20/StatefulPartitionedCallб(encoder_block_21/StatefulPartitionedCallб(encoder_block_22/StatefulPartitionedCallб(encoder_block_23/StatefulPartitionedCallб"sampling_4/StatefulPartitionedCallЋ
(encoder_block_20/StatefulPartitionedCallStatefulPartitionedCalltensor_inputencoder_block_20_1911201encoder_block_20_1911203encoder_block_20_1911205encoder_block_20_1911207encoder_block_20_1911209encoder_block_20_1911211*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1910794║
(encoder_block_21/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_20/StatefulPartitionedCall:output:0encoder_block_21_1911214encoder_block_21_1911216encoder_block_21_1911218encoder_block_21_1911220encoder_block_21_1911222encoder_block_21_1911224*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1910834║
(encoder_block_22/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_21/StatefulPartitionedCall:output:0encoder_block_22_1911227encoder_block_22_1911229encoder_block_22_1911231encoder_block_22_1911233encoder_block_22_1911235encoder_block_22_1911237*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1910874║
(encoder_block_23/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_22/StatefulPartitionedCall:output:0encoder_block_23_1911240encoder_block_23_1911242encoder_block_23_1911244encoder_block_23_1911246encoder_block_23_1911248encoder_block_23_1911250*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1910914»
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_23/StatefulPartitionedCall:output:0conv2d_49_1911253conv2d_49_1911255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1910939т
flatten_4/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_1910951Ћ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_1911259dense_12_1911261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_1910963Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_13_1911264dense_13_1911266*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1910979б
"sampling_4/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_sampling_4_layer_call_and_return_conditional_losses_1911011y
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ{

Identity_1Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ}

Identity_2Identity+sampling_4/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђЂ
NoOpNoOp"^conv2d_49/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall)^encoder_block_20/StatefulPartitionedCall)^encoder_block_21/StatefulPartitionedCall)^encoder_block_22/StatefulPartitionedCall)^encoder_block_23/StatefulPartitionedCall#^sampling_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2T
(encoder_block_20/StatefulPartitionedCall(encoder_block_20/StatefulPartitionedCall2T
(encoder_block_21/StatefulPartitionedCall(encoder_block_21/StatefulPartitionedCall2T
(encoder_block_22/StatefulPartitionedCall(encoder_block_22/StatefulPartitionedCall2T
(encoder_block_23/StatefulPartitionedCall(encoder_block_23/StatefulPartitionedCall2H
"sampling_4/StatefulPartitionedCall"sampling_4/StatefulPartitionedCall:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nametensor_input
┴
N
2__inference_max_pooling2d_20_layer_call_fn_1912736

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
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1910466Ѓ
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
д
t
G__inference_sampling_4_layer_call_and_return_conditional_losses_1911011

inputs
inputs_1
identityѕI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
Shape_1Shapeinputs*
T0*
_output_shapes
::ь¤_
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
 *  ђ?и
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seed2ђЩЧ*
seed▒ т)Ќ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:         ђ}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:         ђJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulmul/x:output:0inputs_1*
T0*(
_output_shapes
:         ђF
ExpExpmul:z:0*
T0*(
_output_shapes
:         ђ[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:         ђR
addAddV2inputs	mul_1:z:0*
T0*(
_output_shapes
:         ђP
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ:         ђ:PL
(
_output_shapes
:         ђ
 
_user_specified_nameinputs:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╝	
ў
2__inference_encoder_block_22_layer_call_fn_1912475
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1910874x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:           ђ
&
_user_specified_nameinput_tensor
─<
г
D__inference_encoder_layer_call_and_return_conditional_losses_1911417
tensor_input3
encoder_block_20_1911345:ђ'
encoder_block_20_1911347:	ђ'
encoder_block_20_1911349:	ђ'
encoder_block_20_1911351:	ђ'
encoder_block_20_1911353:	ђ'
encoder_block_20_1911355:	ђ4
encoder_block_21_1911358:ђђ'
encoder_block_21_1911360:	ђ'
encoder_block_21_1911362:	ђ'
encoder_block_21_1911364:	ђ'
encoder_block_21_1911366:	ђ'
encoder_block_21_1911368:	ђ4
encoder_block_22_1911371:ђђ'
encoder_block_22_1911373:	ђ'
encoder_block_22_1911375:	ђ'
encoder_block_22_1911377:	ђ'
encoder_block_22_1911379:	ђ'
encoder_block_22_1911381:	ђ4
encoder_block_23_1911384:ђђ'
encoder_block_23_1911386:	ђ'
encoder_block_23_1911388:	ђ'
encoder_block_23_1911390:	ђ'
encoder_block_23_1911392:	ђ'
encoder_block_23_1911394:	ђ,
conv2d_49_1911397:ђ
conv2d_49_1911399:$
dense_12_1911403:
ђђ
dense_12_1911405:	ђ$
dense_13_1911408:
ђђ
dense_13_1911410:	ђ
identity

identity_1

identity_2ѕб!conv2d_49/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб(encoder_block_20/StatefulPartitionedCallб(encoder_block_21/StatefulPartitionedCallб(encoder_block_22/StatefulPartitionedCallб(encoder_block_23/StatefulPartitionedCallб"sampling_4/StatefulPartitionedCallЌ
(encoder_block_20/StatefulPartitionedCallStatefulPartitionedCalltensor_inputencoder_block_20_1911345encoder_block_20_1911347encoder_block_20_1911349encoder_block_20_1911351encoder_block_20_1911353encoder_block_20_1911355*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1911044╝
(encoder_block_21/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_20/StatefulPartitionedCall:output:0encoder_block_21_1911358encoder_block_21_1911360encoder_block_21_1911362encoder_block_21_1911364encoder_block_21_1911366encoder_block_21_1911368*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1911083╝
(encoder_block_22/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_21/StatefulPartitionedCall:output:0encoder_block_22_1911371encoder_block_22_1911373encoder_block_22_1911375encoder_block_22_1911377encoder_block_22_1911379encoder_block_22_1911381*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1911122╝
(encoder_block_23/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_22/StatefulPartitionedCall:output:0encoder_block_23_1911384encoder_block_23_1911386encoder_block_23_1911388encoder_block_23_1911390encoder_block_23_1911392encoder_block_23_1911394*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1911161»
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_23/StatefulPartitionedCall:output:0conv2d_49_1911397conv2d_49_1911399*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1910939т
flatten_4/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_1910951Ћ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_1911403dense_12_1911405*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_1910963Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_13_1911408dense_13_1911410*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1910979б
"sampling_4/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_sampling_4_layer_call_and_return_conditional_losses_1911011y
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ{

Identity_1Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ}

Identity_2Identity+sampling_4/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђЂ
NoOpNoOp"^conv2d_49/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall)^encoder_block_20/StatefulPartitionedCall)^encoder_block_21/StatefulPartitionedCall)^encoder_block_22/StatefulPartitionedCall)^encoder_block_23/StatefulPartitionedCall#^sampling_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2T
(encoder_block_20/StatefulPartitionedCall(encoder_block_20/StatefulPartitionedCall2T
(encoder_block_21/StatefulPartitionedCall(encoder_block_21/StatefulPartitionedCall2T
(encoder_block_22/StatefulPartitionedCall(encoder_block_22/StatefulPartitionedCall2T
(encoder_block_23/StatefulPartitionedCall(encoder_block_23/StatefulPartitionedCall2H
"sampling_4/StatefulPartitionedCall"sampling_4/StatefulPartitionedCall:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nametensor_input
й	
Ќ
2__inference_encoder_block_20_layer_call_fn_1912303
input_tensor"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1910794x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         @@ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nameinput_tensor
╚
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_1912661

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
М	
щ
E__inference_dense_12_layer_call_and_return_conditional_losses_1910963

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
а	
О
8__inference_batch_normalization_33_layer_call_fn_1912826

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1910567і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ѕ
ђ
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1910939

inputs9
conv2d_readvariableop_resource:ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
а'
ы
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1910914
input_tensorD
(conv2d_47_conv2d_readvariableop_resource:ђђ8
)conv2d_47_biasadd_readvariableop_resource:	ђ=
.batch_normalization_35_readvariableop_resource:	ђ?
0batch_normalization_35_readvariableop_1_resource:	ђN
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_35/AssignNewValueб'batch_normalization_35/AssignNewValue_1б6batch_normalization_35/FusedBatchNormV3/ReadVariableOpб8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_35/ReadVariableOpб'batch_normalization_35/ReadVariableOp_1б conv2d_47/BiasAdd/ReadVariableOpбconv2d_47/Conv2D/ReadVariableOpњ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_47/Conv2DConv2Dinput_tensor'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Є
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђm
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ»
max_pooling2d_23/MaxPoolMaxPoolconv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_23/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_35/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђБ
NoOpNoOp&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
я
б
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912875

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
а'
ы
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912432
input_tensorD
(conv2d_45_conv2d_readvariableop_resource:ђђ8
)conv2d_45_biasadd_readvariableop_resource:	ђ=
.batch_normalization_33_readvariableop_resource:	ђ?
0batch_normalization_33_readvariableop_1_resource:	ђN
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_33/AssignNewValueб'batch_normalization_33/AssignNewValue_1б6batch_normalization_33/FusedBatchNormV3/ReadVariableOpб8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_33/ReadVariableOpб'batch_normalization_33/ReadVariableOp_1б conv2d_45/BiasAdd/ReadVariableOpбconv2d_45/Conv2D/ReadVariableOpњ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_45/Conv2DConv2Dinput_tensor'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
Є
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђm
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђ»
max_pooling2d_21/MaxPoolMaxPoolconv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_21/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_33/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           ђБ
NoOpNoOp&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@ђ: : : : : : 2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@ђ
&
_user_specified_nameinput_tensor
а	
О
8__inference_batch_normalization_35_layer_call_fn_1912970

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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1910719і
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
Ћ
i
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1910542

inputs
identityб
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
ў
к
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912857

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ќв
ы$
D__inference_encoder_layer_call_and_return_conditional_losses_1912150
tensor_inputT
9encoder_block_20_conv2d_44_conv2d_readvariableop_resource:ђI
:encoder_block_20_conv2d_44_biasadd_readvariableop_resource:	ђN
?encoder_block_20_batch_normalization_32_readvariableop_resource:	ђP
Aencoder_block_20_batch_normalization_32_readvariableop_1_resource:	ђ_
Pencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђU
9encoder_block_21_conv2d_45_conv2d_readvariableop_resource:ђђI
:encoder_block_21_conv2d_45_biasadd_readvariableop_resource:	ђN
?encoder_block_21_batch_normalization_33_readvariableop_resource:	ђP
Aencoder_block_21_batch_normalization_33_readvariableop_1_resource:	ђ_
Pencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђU
9encoder_block_22_conv2d_46_conv2d_readvariableop_resource:ђђI
:encoder_block_22_conv2d_46_biasadd_readvariableop_resource:	ђN
?encoder_block_22_batch_normalization_34_readvariableop_resource:	ђP
Aencoder_block_22_batch_normalization_34_readvariableop_1_resource:	ђ_
Pencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђU
9encoder_block_23_conv2d_47_conv2d_readvariableop_resource:ђђI
:encoder_block_23_conv2d_47_biasadd_readvariableop_resource:	ђN
?encoder_block_23_batch_normalization_35_readvariableop_resource:	ђP
Aencoder_block_23_batch_normalization_35_readvariableop_1_resource:	ђ_
Pencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђC
(conv2d_49_conv2d_readvariableop_resource:ђ7
)conv2d_49_biasadd_readvariableop_resource:;
'dense_12_matmul_readvariableop_resource:
ђђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕб conv2d_49/BiasAdd/ReadVariableOpбconv2d_49/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpб6encoder_block_20/batch_normalization_32/AssignNewValueб8encoder_block_20/batch_normalization_32/AssignNewValue_1бGencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpбIencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_20/batch_normalization_32/ReadVariableOpб8encoder_block_20/batch_normalization_32/ReadVariableOp_1б1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOpб0encoder_block_20/conv2d_44/Conv2D/ReadVariableOpб6encoder_block_21/batch_normalization_33/AssignNewValueб8encoder_block_21/batch_normalization_33/AssignNewValue_1бGencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpбIencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_21/batch_normalization_33/ReadVariableOpб8encoder_block_21/batch_normalization_33/ReadVariableOp_1б1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOpб0encoder_block_21/conv2d_45/Conv2D/ReadVariableOpб6encoder_block_22/batch_normalization_34/AssignNewValueб8encoder_block_22/batch_normalization_34/AssignNewValue_1бGencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpбIencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_22/batch_normalization_34/ReadVariableOpб8encoder_block_22/batch_normalization_34/ReadVariableOp_1б1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOpб0encoder_block_22/conv2d_46/Conv2D/ReadVariableOpб6encoder_block_23/batch_normalization_35/AssignNewValueб8encoder_block_23/batch_normalization_35/AssignNewValue_1бGencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpбIencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_23/batch_normalization_35/ReadVariableOpб8encoder_block_23/batch_normalization_35/ReadVariableOp_1б1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOpб0encoder_block_23/conv2d_47/Conv2D/ReadVariableOp│
0encoder_block_20/conv2d_44/Conv2D/ReadVariableOpReadVariableOp9encoder_block_20_conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0п
!encoder_block_20/conv2d_44/Conv2DConv2Dtensor_input8encoder_block_20/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
Е
1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_20_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Л
"encoder_block_20/conv2d_44/BiasAddBiasAdd*encoder_block_20/conv2d_44/Conv2D:output:09encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђЉ
encoder_block_20/conv2d_44/ReluRelu+encoder_block_20/conv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђЛ
)encoder_block_20/max_pooling2d_20/MaxPoolMaxPool-encoder_block_20/conv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_20/batch_normalization_32/ReadVariableOpReadVariableOp?encoder_block_20_batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_20/batch_normalization_32/ReadVariableOp_1ReadVariableOpAencoder_block_20_batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
8encoder_block_20/batch_normalization_32/FusedBatchNormV3FusedBatchNormV32encoder_block_20/max_pooling2d_20/MaxPool:output:0>encoder_block_20/batch_normalization_32/ReadVariableOp:value:0@encoder_block_20/batch_normalization_32/ReadVariableOp_1:value:0Oencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6encoder_block_20/batch_normalization_32/AssignNewValueAssignVariableOpPencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resourceEencoder_block_20/batch_normalization_32/FusedBatchNormV3:batch_mean:0H^encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8encoder_block_20/batch_normalization_32/AssignNewValue_1AssignVariableOpRencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_20/batch_normalization_32/FusedBatchNormV3:batch_variance:0J^encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
0encoder_block_21/conv2d_45/Conv2D/ReadVariableOpReadVariableOp9encoder_block_21_conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0є
!encoder_block_21/conv2d_45/Conv2DConv2D<encoder_block_20/batch_normalization_32/FusedBatchNormV3:y:08encoder_block_21/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
Е
1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_21_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"encoder_block_21/conv2d_45/BiasAddBiasAdd*encoder_block_21/conv2d_45/Conv2D:output:09encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђЈ
encoder_block_21/conv2d_45/ReluRelu+encoder_block_21/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђЛ
)encoder_block_21/max_pooling2d_21/MaxPoolMaxPool-encoder_block_21/conv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_21/batch_normalization_33/ReadVariableOpReadVariableOp?encoder_block_21_batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_21/batch_normalization_33/ReadVariableOp_1ReadVariableOpAencoder_block_21_batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
8encoder_block_21/batch_normalization_33/FusedBatchNormV3FusedBatchNormV32encoder_block_21/max_pooling2d_21/MaxPool:output:0>encoder_block_21/batch_normalization_33/ReadVariableOp:value:0@encoder_block_21/batch_normalization_33/ReadVariableOp_1:value:0Oencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6encoder_block_21/batch_normalization_33/AssignNewValueAssignVariableOpPencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resourceEencoder_block_21/batch_normalization_33/FusedBatchNormV3:batch_mean:0H^encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8encoder_block_21/batch_normalization_33/AssignNewValue_1AssignVariableOpRencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_21/batch_normalization_33/FusedBatchNormV3:batch_variance:0J^encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
0encoder_block_22/conv2d_46/Conv2D/ReadVariableOpReadVariableOp9encoder_block_22_conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0є
!encoder_block_22/conv2d_46/Conv2DConv2D<encoder_block_21/batch_normalization_33/FusedBatchNormV3:y:08encoder_block_22/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Е
1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_22_conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"encoder_block_22/conv2d_46/BiasAddBiasAdd*encoder_block_22/conv2d_46/Conv2D:output:09encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђЈ
encoder_block_22/conv2d_46/ReluRelu+encoder_block_22/conv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђЛ
)encoder_block_22/max_pooling2d_22/MaxPoolMaxPool-encoder_block_22/conv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_22/batch_normalization_34/ReadVariableOpReadVariableOp?encoder_block_22_batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_22/batch_normalization_34/ReadVariableOp_1ReadVariableOpAencoder_block_22_batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
8encoder_block_22/batch_normalization_34/FusedBatchNormV3FusedBatchNormV32encoder_block_22/max_pooling2d_22/MaxPool:output:0>encoder_block_22/batch_normalization_34/ReadVariableOp:value:0@encoder_block_22/batch_normalization_34/ReadVariableOp_1:value:0Oencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6encoder_block_22/batch_normalization_34/AssignNewValueAssignVariableOpPencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resourceEencoder_block_22/batch_normalization_34/FusedBatchNormV3:batch_mean:0H^encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8encoder_block_22/batch_normalization_34/AssignNewValue_1AssignVariableOpRencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_22/batch_normalization_34/FusedBatchNormV3:batch_variance:0J^encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
0encoder_block_23/conv2d_47/Conv2D/ReadVariableOpReadVariableOp9encoder_block_23_conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0є
!encoder_block_23/conv2d_47/Conv2DConv2D<encoder_block_22/batch_normalization_34/FusedBatchNormV3:y:08encoder_block_23/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Е
1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_23_conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"encoder_block_23/conv2d_47/BiasAddBiasAdd*encoder_block_23/conv2d_47/Conv2D:output:09encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЈ
encoder_block_23/conv2d_47/ReluRelu+encoder_block_23/conv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђЛ
)encoder_block_23/max_pooling2d_23/MaxPoolMaxPool-encoder_block_23/conv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_23/batch_normalization_35/ReadVariableOpReadVariableOp?encoder_block_23_batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_23/batch_normalization_35/ReadVariableOp_1ReadVariableOpAencoder_block_23_batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0й
8encoder_block_23/batch_normalization_35/FusedBatchNormV3FusedBatchNormV32encoder_block_23/max_pooling2d_23/MaxPool:output:0>encoder_block_23/batch_normalization_35/ReadVariableOp:value:0@encoder_block_23/batch_normalization_35/ReadVariableOp_1:value:0Oencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6encoder_block_23/batch_normalization_35/AssignNewValueAssignVariableOpPencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resourceEencoder_block_23/batch_normalization_35/FusedBatchNormV3:batch_mean:0H^encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8encoder_block_23/batch_normalization_35/AssignNewValue_1AssignVariableOpRencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resourceIencoder_block_23/batch_normalization_35/FusedBatchNormV3:batch_variance:0J^encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Љ
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0с
conv2d_49/Conv2DConv2D<encoder_block_23/batch_normalization_35/FusedBatchNormV3:y:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:         `
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Є
flatten_4/ReshapeReshapeconv2d_49/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђѕ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0љ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЁ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0њ
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђѕ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0љ
dense_13/MatMulMatMulflatten_4/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЁ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0њ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђg
sampling_4/ShapeShapedense_12/BiasAdd:output:0*
T0*
_output_shapes
::ь¤h
sampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
sampling_4/strided_sliceStridedSlicesampling_4/Shape:output:0'sampling_4/strided_slice/stack:output:0)sampling_4/strided_slice/stack_1:output:0)sampling_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
sampling_4/Shape_1Shapedense_12/BiasAdd:output:0*
T0*
_output_shapes
::ь¤j
 sampling_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
sampling_4/strided_slice_1StridedSlicesampling_4/Shape_1:output:0)sampling_4/strided_slice_1/stack:output:0+sampling_4/strided_slice_1/stack_1:output:0+sampling_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskю
sampling_4/random_normal/shapePack!sampling_4/strided_slice:output:0#sampling_4/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?═
-sampling_4/random_normal/RandomStandardNormalRandomStandardNormal'sampling_4/random_normal/shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seed2Хсџ*
seed▒ т)И
sampling_4/random_normal/mulMul6sampling_4/random_normal/RandomStandardNormal:output:0(sampling_4/random_normal/stddev:output:0*
T0*(
_output_shapes
:         ђъ
sampling_4/random_normalAddV2 sampling_4/random_normal/mul:z:0&sampling_4/random_normal/mean:output:0*
T0*(
_output_shapes
:         ђU
sampling_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
sampling_4/mulMulsampling_4/mul/x:output:0dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ\
sampling_4/ExpExpsampling_4/mul:z:0*
T0*(
_output_shapes
:         ђ|
sampling_4/mul_1Mulsampling_4/Exp:y:0sampling_4/random_normal:z:0*
T0*(
_output_shapes
:         ђ{
sampling_4/addAddV2dense_12/BiasAdd:output:0sampling_4/mul_1:z:0*
T0*(
_output_shapes
:         ђi
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђk

Identity_1Identitydense_13/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђd

Identity_2Identitysampling_4/add:z:0^NoOp*
T0*(
_output_shapes
:         ђЦ
NoOpNoOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp7^encoder_block_20/batch_normalization_32/AssignNewValue9^encoder_block_20/batch_normalization_32/AssignNewValue_1H^encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpJ^encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_17^encoder_block_20/batch_normalization_32/ReadVariableOp9^encoder_block_20/batch_normalization_32/ReadVariableOp_12^encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp1^encoder_block_20/conv2d_44/Conv2D/ReadVariableOp7^encoder_block_21/batch_normalization_33/AssignNewValue9^encoder_block_21/batch_normalization_33/AssignNewValue_1H^encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpJ^encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_17^encoder_block_21/batch_normalization_33/ReadVariableOp9^encoder_block_21/batch_normalization_33/ReadVariableOp_12^encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp1^encoder_block_21/conv2d_45/Conv2D/ReadVariableOp7^encoder_block_22/batch_normalization_34/AssignNewValue9^encoder_block_22/batch_normalization_34/AssignNewValue_1H^encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpJ^encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_17^encoder_block_22/batch_normalization_34/ReadVariableOp9^encoder_block_22/batch_normalization_34/ReadVariableOp_12^encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp1^encoder_block_22/conv2d_46/Conv2D/ReadVariableOp7^encoder_block_23/batch_normalization_35/AssignNewValue9^encoder_block_23/batch_normalization_35/AssignNewValue_1H^encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpJ^encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_17^encoder_block_23/batch_normalization_35/ReadVariableOp9^encoder_block_23/batch_normalization_35/ReadVariableOp_12^encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp1^encoder_block_23/conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2t
8encoder_block_20/batch_normalization_32/AssignNewValue_18encoder_block_20/batch_normalization_32/AssignNewValue_12p
6encoder_block_20/batch_normalization_32/AssignNewValue6encoder_block_20/batch_normalization_32/AssignNewValue2ќ
Iencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpGencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_20/batch_normalization_32/ReadVariableOp_18encoder_block_20/batch_normalization_32/ReadVariableOp_12p
6encoder_block_20/batch_normalization_32/ReadVariableOp6encoder_block_20/batch_normalization_32/ReadVariableOp2f
1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp2d
0encoder_block_20/conv2d_44/Conv2D/ReadVariableOp0encoder_block_20/conv2d_44/Conv2D/ReadVariableOp2t
8encoder_block_21/batch_normalization_33/AssignNewValue_18encoder_block_21/batch_normalization_33/AssignNewValue_12p
6encoder_block_21/batch_normalization_33/AssignNewValue6encoder_block_21/batch_normalization_33/AssignNewValue2ќ
Iencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpGencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_21/batch_normalization_33/ReadVariableOp_18encoder_block_21/batch_normalization_33/ReadVariableOp_12p
6encoder_block_21/batch_normalization_33/ReadVariableOp6encoder_block_21/batch_normalization_33/ReadVariableOp2f
1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp2d
0encoder_block_21/conv2d_45/Conv2D/ReadVariableOp0encoder_block_21/conv2d_45/Conv2D/ReadVariableOp2t
8encoder_block_22/batch_normalization_34/AssignNewValue_18encoder_block_22/batch_normalization_34/AssignNewValue_12p
6encoder_block_22/batch_normalization_34/AssignNewValue6encoder_block_22/batch_normalization_34/AssignNewValue2ќ
Iencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpGencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_22/batch_normalization_34/ReadVariableOp_18encoder_block_22/batch_normalization_34/ReadVariableOp_12p
6encoder_block_22/batch_normalization_34/ReadVariableOp6encoder_block_22/batch_normalization_34/ReadVariableOp2f
1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp2d
0encoder_block_22/conv2d_46/Conv2D/ReadVariableOp0encoder_block_22/conv2d_46/Conv2D/ReadVariableOp2t
8encoder_block_23/batch_normalization_35/AssignNewValue_18encoder_block_23/batch_normalization_35/AssignNewValue_12p
6encoder_block_23/batch_normalization_35/AssignNewValue6encoder_block_23/batch_normalization_35/AssignNewValue2ќ
Iencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpGencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_23/batch_normalization_35/ReadVariableOp_18encoder_block_23/batch_normalization_35/ReadVariableOp_12p
6encoder_block_23/batch_normalization_35/ReadVariableOp6encoder_block_23/batch_normalization_35/ReadVariableOp2f
1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp2d
0encoder_block_23/conv2d_47/Conv2D/ReadVariableOp0encoder_block_23/conv2d_47/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nametensor_input
К
ь
)__inference_encoder_layer_call_fn_1911945
tensor_input"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ%

unknown_23:ђ

unknown_24:

unknown_25:
ђђ

unknown_26:	ђ

unknown_27:
ђђ

unknown_28:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallє
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
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1911273p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nametensor_input
Ч
ъ
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912372
input_tensorC
(conv2d_44_conv2d_readvariableop_resource:ђ8
)conv2d_44_biasadd_readvariableop_resource:	ђ=
.batch_normalization_32_readvariableop_resource:	ђ?
0batch_normalization_32_readvariableop_1_resource:	ђN
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_32/FusedBatchNormV3/ReadVariableOpб8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_32/ReadVariableOpб'batch_normalization_32/ReadVariableOp_1б conv2d_44/BiasAdd/ReadVariableOpбconv2d_44/Conv2D/ReadVariableOpЉ
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Х
conv2d_44/Conv2DConv2Dinput_tensor'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
Є
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ъ
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђo
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђ»
max_pooling2d_20/MaxPoolMaxPoolconv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_20/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_32/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@ђЛ
NoOpNoOp7^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nameinput_tensor
д'
­
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1910794
input_tensorC
(conv2d_44_conv2d_readvariableop_resource:ђ8
)conv2d_44_biasadd_readvariableop_resource:	ђ=
.batch_normalization_32_readvariableop_resource:	ђ?
0batch_normalization_32_readvariableop_1_resource:	ђN
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_32/AssignNewValueб'batch_normalization_32/AssignNewValue_1б6batch_normalization_32/FusedBatchNormV3/ReadVariableOpб8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_32/ReadVariableOpб'batch_normalization_32/ReadVariableOp_1б conv2d_44/BiasAdd/ReadVariableOpбconv2d_44/Conv2D/ReadVariableOpЉ
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Х
conv2d_44/Conv2DConv2Dinput_tensor'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
Є
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ъ
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђo
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђ»
max_pooling2d_20/MaxPoolMaxPoolconv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_20/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_32/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@ђБ
NoOpNoOp&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nameinput_tensor
└
У
)__inference_encoder_layer_call_fn_1911484
input_1"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ%

unknown_23:ђ

unknown_24:

unknown_25:
ђђ

unknown_26:	ђ

unknown_27:
ђђ

unknown_28:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЅ
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
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1911417p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
М	
щ
E__inference_dense_13_layer_call_and_return_conditional_losses_1910979

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╚
b
F__inference_flatten_4_layer_call_and_return_conditional_losses_1910951

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
М	
щ
E__inference_dense_12_layer_call_and_return_conditional_losses_1912680

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ѕ
ђ
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1912650

inputs9
conv2d_readvariableop_resource:ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Ў
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
А
+__inference_conv2d_49_layer_call_fn_1912639

inputs"
unknown:ђ
	unknown_0:
identityѕбStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1910939w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
╦і
Х
#__inference__traced_restore_1913324
file_prefixU
:assignvariableop_encoder_encoder_block_20_conv2d_44_kernel:ђI
:assignvariableop_1_encoder_encoder_block_20_conv2d_44_bias:	ђW
Hassignvariableop_2_encoder_encoder_block_20_batch_normalization_32_gamma:	ђV
Gassignvariableop_3_encoder_encoder_block_20_batch_normalization_32_beta:	ђ]
Nassignvariableop_4_encoder_encoder_block_20_batch_normalization_32_moving_mean:	ђa
Rassignvariableop_5_encoder_encoder_block_20_batch_normalization_32_moving_variance:	ђX
<assignvariableop_6_encoder_encoder_block_21_conv2d_45_kernel:ђђI
:assignvariableop_7_encoder_encoder_block_21_conv2d_45_bias:	ђW
Hassignvariableop_8_encoder_encoder_block_21_batch_normalization_33_gamma:	ђV
Gassignvariableop_9_encoder_encoder_block_21_batch_normalization_33_beta:	ђ^
Oassignvariableop_10_encoder_encoder_block_21_batch_normalization_33_moving_mean:	ђb
Sassignvariableop_11_encoder_encoder_block_21_batch_normalization_33_moving_variance:	ђY
=assignvariableop_12_encoder_encoder_block_22_conv2d_46_kernel:ђђJ
;assignvariableop_13_encoder_encoder_block_22_conv2d_46_bias:	ђX
Iassignvariableop_14_encoder_encoder_block_22_batch_normalization_34_gamma:	ђW
Hassignvariableop_15_encoder_encoder_block_22_batch_normalization_34_beta:	ђ^
Oassignvariableop_16_encoder_encoder_block_22_batch_normalization_34_moving_mean:	ђb
Sassignvariableop_17_encoder_encoder_block_22_batch_normalization_34_moving_variance:	ђY
=assignvariableop_18_encoder_encoder_block_23_conv2d_47_kernel:ђђJ
;assignvariableop_19_encoder_encoder_block_23_conv2d_47_bias:	ђX
Iassignvariableop_20_encoder_encoder_block_23_batch_normalization_35_gamma:	ђW
Hassignvariableop_21_encoder_encoder_block_23_batch_normalization_35_beta:	ђ^
Oassignvariableop_22_encoder_encoder_block_23_batch_normalization_35_moving_mean:	ђb
Sassignvariableop_23_encoder_encoder_block_23_batch_normalization_35_moving_variance:	ђG
,assignvariableop_24_encoder_conv2d_49_kernel:ђ8
*assignvariableop_25_encoder_conv2d_49_bias:?
+assignvariableop_26_encoder_dense_12_kernel:
ђђ8
)assignvariableop_27_encoder_dense_12_bias:	ђ?
+assignvariableop_28_encoder_dense_13_kernel:
ђђ8
)assignvariableop_29_encoder_dense_13_bias:	ђ
identity_31ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Л

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*э	
valueь	BЖ	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH«
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ║
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*љ
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:═
AssignVariableOpAssignVariableOp:assignvariableop_encoder_encoder_block_20_conv2d_44_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_1AssignVariableOp:assignvariableop_1_encoder_encoder_block_20_conv2d_44_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_2AssignVariableOpHassignvariableop_2_encoder_encoder_block_20_batch_normalization_32_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:я
AssignVariableOp_3AssignVariableOpGassignvariableop_3_encoder_encoder_block_20_batch_normalization_32_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:т
AssignVariableOp_4AssignVariableOpNassignvariableop_4_encoder_encoder_block_20_batch_normalization_32_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_5AssignVariableOpRassignvariableop_5_encoder_encoder_block_20_batch_normalization_32_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_6AssignVariableOp<assignvariableop_6_encoder_encoder_block_21_conv2d_45_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_7AssignVariableOp:assignvariableop_7_encoder_encoder_block_21_conv2d_45_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_8AssignVariableOpHassignvariableop_8_encoder_encoder_block_21_batch_normalization_33_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:я
AssignVariableOp_9AssignVariableOpGassignvariableop_9_encoder_encoder_block_21_batch_normalization_33_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_10AssignVariableOpOassignvariableop_10_encoder_encoder_block_21_batch_normalization_33_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_11AssignVariableOpSassignvariableop_11_encoder_encoder_block_21_batch_normalization_33_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_12AssignVariableOp=assignvariableop_12_encoder_encoder_block_22_conv2d_46_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_13AssignVariableOp;assignvariableop_13_encoder_encoder_block_22_conv2d_46_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_14AssignVariableOpIassignvariableop_14_encoder_encoder_block_22_batch_normalization_34_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_15AssignVariableOpHassignvariableop_15_encoder_encoder_block_22_batch_normalization_34_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_16AssignVariableOpOassignvariableop_16_encoder_encoder_block_22_batch_normalization_34_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_17AssignVariableOpSassignvariableop_17_encoder_encoder_block_22_batch_normalization_34_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOp_18AssignVariableOp=assignvariableop_18_encoder_encoder_block_23_conv2d_47_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_19AssignVariableOp;assignvariableop_19_encoder_encoder_block_23_conv2d_47_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_20AssignVariableOpIassignvariableop_20_encoder_encoder_block_23_batch_normalization_35_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:р
AssignVariableOp_21AssignVariableOpHassignvariableop_21_encoder_encoder_block_23_batch_normalization_35_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOpOassignvariableop_22_encoder_encoder_block_23_batch_normalization_35_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_23AssignVariableOpSassignvariableop_23_encoder_encoder_block_23_batch_normalization_35_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:┼
AssignVariableOp_24AssignVariableOp,assignvariableop_24_encoder_conv2d_49_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:├
AssignVariableOp_25AssignVariableOp*assignvariableop_25_encoder_conv2d_49_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_26AssignVariableOp+assignvariableop_26_encoder_dense_12_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_27AssignVariableOp)assignvariableop_27_encoder_dense_12_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_28AssignVariableOp+assignvariableop_28_encoder_dense_13_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_29AssignVariableOp)assignvariableop_29_encoder_dense_13_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 с
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: л
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
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
я
б
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913019

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
х<
Д
D__inference_encoder_layer_call_and_return_conditional_losses_1911195
input_13
encoder_block_20_1911045:ђ'
encoder_block_20_1911047:	ђ'
encoder_block_20_1911049:	ђ'
encoder_block_20_1911051:	ђ'
encoder_block_20_1911053:	ђ'
encoder_block_20_1911055:	ђ4
encoder_block_21_1911084:ђђ'
encoder_block_21_1911086:	ђ'
encoder_block_21_1911088:	ђ'
encoder_block_21_1911090:	ђ'
encoder_block_21_1911092:	ђ'
encoder_block_21_1911094:	ђ4
encoder_block_22_1911123:ђђ'
encoder_block_22_1911125:	ђ'
encoder_block_22_1911127:	ђ'
encoder_block_22_1911129:	ђ'
encoder_block_22_1911131:	ђ'
encoder_block_22_1911133:	ђ4
encoder_block_23_1911162:ђђ'
encoder_block_23_1911164:	ђ'
encoder_block_23_1911166:	ђ'
encoder_block_23_1911168:	ђ'
encoder_block_23_1911170:	ђ'
encoder_block_23_1911172:	ђ,
conv2d_49_1911175:ђ
conv2d_49_1911177:$
dense_12_1911181:
ђђ
dense_12_1911183:	ђ$
dense_13_1911186:
ђђ
dense_13_1911188:	ђ
identity

identity_1

identity_2ѕб!conv2d_49/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб(encoder_block_20/StatefulPartitionedCallб(encoder_block_21/StatefulPartitionedCallб(encoder_block_22/StatefulPartitionedCallб(encoder_block_23/StatefulPartitionedCallб"sampling_4/StatefulPartitionedCallњ
(encoder_block_20/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_block_20_1911045encoder_block_20_1911047encoder_block_20_1911049encoder_block_20_1911051encoder_block_20_1911053encoder_block_20_1911055*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1911044╝
(encoder_block_21/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_20/StatefulPartitionedCall:output:0encoder_block_21_1911084encoder_block_21_1911086encoder_block_21_1911088encoder_block_21_1911090encoder_block_21_1911092encoder_block_21_1911094*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1911083╝
(encoder_block_22/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_21/StatefulPartitionedCall:output:0encoder_block_22_1911123encoder_block_22_1911125encoder_block_22_1911127encoder_block_22_1911129encoder_block_22_1911131encoder_block_22_1911133*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1911122╝
(encoder_block_23/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_22/StatefulPartitionedCall:output:0encoder_block_23_1911162encoder_block_23_1911164encoder_block_23_1911166encoder_block_23_1911168encoder_block_23_1911170encoder_block_23_1911172*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1911161»
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_23/StatefulPartitionedCall:output:0conv2d_49_1911175conv2d_49_1911177*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1910939т
flatten_4/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_1910951Ћ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_1911181dense_12_1911183*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_1910963Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_13_1911186dense_13_1911188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1910979б
"sampling_4/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_sampling_4_layer_call_and_return_conditional_losses_1911011y
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ{

Identity_1Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ}

Identity_2Identity+sampling_4/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђЂ
NoOpNoOp"^conv2d_49/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall)^encoder_block_20/StatefulPartitionedCall)^encoder_block_21/StatefulPartitionedCall)^encoder_block_22/StatefulPartitionedCall)^encoder_block_23/StatefulPartitionedCall#^sampling_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2T
(encoder_block_20/StatefulPartitionedCall(encoder_block_20/StatefulPartitionedCall2T
(encoder_block_21/StatefulPartitionedCall(encoder_block_21/StatefulPartitionedCall2T
(encoder_block_22/StatefulPartitionedCall(encoder_block_22/StatefulPartitionedCall2T
(encoder_block_23/StatefulPartitionedCall(encoder_block_23/StatefulPartitionedCall2H
"sampling_4/StatefulPartitionedCall"sampling_4/StatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
я
б
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1910585

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
д'
­
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912346
input_tensorC
(conv2d_44_conv2d_readvariableop_resource:ђ8
)conv2d_44_biasadd_readvariableop_resource:	ђ=
.batch_normalization_32_readvariableop_resource:	ђ?
0batch_normalization_32_readvariableop_1_resource:	ђN
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_32/AssignNewValueб'batch_normalization_32/AssignNewValue_1б6batch_normalization_32/FusedBatchNormV3/ReadVariableOpб8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_32/ReadVariableOpб'batch_normalization_32/ReadVariableOp_1б conv2d_44/BiasAdd/ReadVariableOpбconv2d_44/Conv2D/ReadVariableOpЉ
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Х
conv2d_44/Conv2DConv2Dinput_tensor'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
Є
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ъ
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђo
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђ»
max_pooling2d_20/MaxPoolMaxPoolconv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_20/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_32/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@ђБ
NoOpNoOp&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nameinput_tensor
Ш
Ъ
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1911161
input_tensorD
(conv2d_47_conv2d_readvariableop_resource:ђђ8
)conv2d_47_biasadd_readvariableop_resource:	ђ=
.batch_normalization_35_readvariableop_resource:	ђ?
0batch_normalization_35_readvariableop_1_resource:	ђN
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_35/FusedBatchNormV3/ReadVariableOpб8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_35/ReadVariableOpб'batch_normalization_35/ReadVariableOp_1б conv2d_47/BiasAdd/ReadVariableOpбconv2d_47/Conv2D/ReadVariableOpњ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_47/Conv2DConv2Dinput_tensor'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Є
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђm
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ»
max_pooling2d_23/MaxPoolMaxPoolconv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_23/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_35/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђЛ
NoOpNoOp7^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
я
б
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912947

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
я
б
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912803

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╝	
ў
2__inference_encoder_block_21_layer_call_fn_1912389
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1910834x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         @@ђ
&
_user_specified_nameinput_tensor
М	
щ
E__inference_dense_13_layer_call_and_return_conditional_losses_1912699

inputs2
matmul_readvariableop_resource:
ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┴
N
2__inference_max_pooling2d_21_layer_call_fn_1912808

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
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1910542Ѓ
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
Й	
ў
2__inference_encoder_block_23_layer_call_fn_1912578
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1911161x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
Ш
Ъ
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912544
input_tensorD
(conv2d_46_conv2d_readvariableop_resource:ђђ8
)conv2d_46_biasadd_readvariableop_resource:	ђ=
.batch_normalization_34_readvariableop_resource:	ђ?
0batch_normalization_34_readvariableop_1_resource:	ђN
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_34/FusedBatchNormV3/ReadVariableOpб8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_34/ReadVariableOpб'batch_normalization_34/ReadVariableOp_1б conv2d_46/BiasAdd/ReadVariableOpбconv2d_46/Conv2D/ReadVariableOpњ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_46/Conv2DConv2Dinput_tensor'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Є
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђm
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ»
max_pooling2d_22/MaxPoolMaxPoolconv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_22/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_34/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђЛ
NoOpNoOp7^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           ђ: : : : : : 2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           ђ
&
_user_specified_nameinput_tensor
џ
С
%__inference_signature_wrapper_1911876
input_1"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ%

unknown_23:ђ

unknown_24:

unknown_25:
ђђ

unknown_26:	ђ

unknown_27:
ђђ

unknown_28:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallу
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
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *+
f&R$
"__inference__wrapped_model_1910460p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
Й	
ў
2__inference_encoder_block_21_layer_call_fn_1912406
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1911083x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         @@ђ
&
_user_specified_nameinput_tensor
Ш
Ъ
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1911083
input_tensorD
(conv2d_45_conv2d_readvariableop_resource:ђђ8
)conv2d_45_biasadd_readvariableop_resource:	ђ=
.batch_normalization_33_readvariableop_resource:	ђ?
0batch_normalization_33_readvariableop_1_resource:	ђN
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_33/FusedBatchNormV3/ReadVariableOpб8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_33/ReadVariableOpб'batch_normalization_33/ReadVariableOp_1б conv2d_45/BiasAdd/ReadVariableOpбconv2d_45/Conv2D/ReadVariableOpњ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_45/Conv2DConv2Dinput_tensor'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
Є
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђm
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђ»
max_pooling2d_21/MaxPoolMaxPoolconv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_21/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_33/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           ђЛ
NoOpNoOp7^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@ђ: : : : : : 2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@ђ
&
_user_specified_nameinput_tensor
Ћ
i
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1912957

inputs
identityб
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
Г<
Д
D__inference_encoder_layer_call_and_return_conditional_losses_1911016
input_13
encoder_block_20_1910795:ђ'
encoder_block_20_1910797:	ђ'
encoder_block_20_1910799:	ђ'
encoder_block_20_1910801:	ђ'
encoder_block_20_1910803:	ђ'
encoder_block_20_1910805:	ђ4
encoder_block_21_1910835:ђђ'
encoder_block_21_1910837:	ђ'
encoder_block_21_1910839:	ђ'
encoder_block_21_1910841:	ђ'
encoder_block_21_1910843:	ђ'
encoder_block_21_1910845:	ђ4
encoder_block_22_1910875:ђђ'
encoder_block_22_1910877:	ђ'
encoder_block_22_1910879:	ђ'
encoder_block_22_1910881:	ђ'
encoder_block_22_1910883:	ђ'
encoder_block_22_1910885:	ђ4
encoder_block_23_1910915:ђђ'
encoder_block_23_1910917:	ђ'
encoder_block_23_1910919:	ђ'
encoder_block_23_1910921:	ђ'
encoder_block_23_1910923:	ђ'
encoder_block_23_1910925:	ђ,
conv2d_49_1910940:ђ
conv2d_49_1910942:$
dense_12_1910964:
ђђ
dense_12_1910966:	ђ$
dense_13_1910980:
ђђ
dense_13_1910982:	ђ
identity

identity_1

identity_2ѕб!conv2d_49/StatefulPartitionedCallб dense_12/StatefulPartitionedCallб dense_13/StatefulPartitionedCallб(encoder_block_20/StatefulPartitionedCallб(encoder_block_21/StatefulPartitionedCallб(encoder_block_22/StatefulPartitionedCallб(encoder_block_23/StatefulPartitionedCallб"sampling_4/StatefulPartitionedCallљ
(encoder_block_20/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_block_20_1910795encoder_block_20_1910797encoder_block_20_1910799encoder_block_20_1910801encoder_block_20_1910803encoder_block_20_1910805*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1910794║
(encoder_block_21/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_20/StatefulPartitionedCall:output:0encoder_block_21_1910835encoder_block_21_1910837encoder_block_21_1910839encoder_block_21_1910841encoder_block_21_1910843encoder_block_21_1910845*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:           ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1910834║
(encoder_block_22/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_21/StatefulPartitionedCall:output:0encoder_block_22_1910875encoder_block_22_1910877encoder_block_22_1910879encoder_block_22_1910881encoder_block_22_1910883encoder_block_22_1910885*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1910874║
(encoder_block_23/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_22/StatefulPartitionedCall:output:0encoder_block_23_1910915encoder_block_23_1910917encoder_block_23_1910919encoder_block_23_1910921encoder_block_23_1910923encoder_block_23_1910925*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1910914»
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall1encoder_block_23/StatefulPartitionedCall:output:0conv2d_49_1910940conv2d_49_1910942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1910939т
flatten_4/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *O
fJRH
F__inference_flatten_4_layer_call_and_return_conditional_losses_1910951Ћ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_12_1910964dense_12_1910966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_1910963Ћ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0dense_13_1910980dense_13_1910982*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1910979б
"sampling_4/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *P
fKRI
G__inference_sampling_4_layer_call_and_return_conditional_losses_1911011y
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ{

Identity_1Identity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ}

Identity_2Identity+sampling_4/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђЂ
NoOpNoOp"^conv2d_49/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall)^encoder_block_20/StatefulPartitionedCall)^encoder_block_21/StatefulPartitionedCall)^encoder_block_22/StatefulPartitionedCall)^encoder_block_23/StatefulPartitionedCall#^sampling_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2T
(encoder_block_20/StatefulPartitionedCall(encoder_block_20/StatefulPartitionedCall2T
(encoder_block_21/StatefulPartitionedCall(encoder_block_21/StatefulPartitionedCall2T
(encoder_block_22/StatefulPartitionedCall(encoder_block_22/StatefulPartitionedCall2T
(encoder_block_23/StatefulPartitionedCall(encoder_block_23/StatefulPartitionedCall2H
"sampling_4/StatefulPartitionedCall"sampling_4/StatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
┴
N
2__inference_max_pooling2d_23_layer_call_fn_1912952

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
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1910694Ѓ
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
ў
к
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912929

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
я
б
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1910509

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
┐	
Ќ
2__inference_encoder_block_20_layer_call_fn_1912320
input_tensor"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         @@ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1911044x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         @@ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nameinput_tensor
¤
ь
)__inference_encoder_layer_call_fn_1912014
tensor_input"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ%

unknown_23:ђ

unknown_24:

unknown_25:
ђђ

unknown_26:	ђ

unknown_27:
ђђ

unknown_28:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallј
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
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*@
_read_only_resource_inputs"
 	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1911417p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nametensor_input
Ћ
i
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1910466

inputs
identityб
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
Ћ
i
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1910694

inputs
identityб
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
Ћ
i
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1912813

inputs
identityб
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
Л┌
┌$
"__inference__wrapped_model_1910460
input_1\
Aencoder_encoder_block_20_conv2d_44_conv2d_readvariableop_resource:ђQ
Bencoder_encoder_block_20_conv2d_44_biasadd_readvariableop_resource:	ђV
Gencoder_encoder_block_20_batch_normalization_32_readvariableop_resource:	ђX
Iencoder_encoder_block_20_batch_normalization_32_readvariableop_1_resource:	ђg
Xencoder_encoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђi
Zencoder_encoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Aencoder_encoder_block_21_conv2d_45_conv2d_readvariableop_resource:ђђQ
Bencoder_encoder_block_21_conv2d_45_biasadd_readvariableop_resource:	ђV
Gencoder_encoder_block_21_batch_normalization_33_readvariableop_resource:	ђX
Iencoder_encoder_block_21_batch_normalization_33_readvariableop_1_resource:	ђg
Xencoder_encoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђi
Zencoder_encoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Aencoder_encoder_block_22_conv2d_46_conv2d_readvariableop_resource:ђђQ
Bencoder_encoder_block_22_conv2d_46_biasadd_readvariableop_resource:	ђV
Gencoder_encoder_block_22_batch_normalization_34_readvariableop_resource:	ђX
Iencoder_encoder_block_22_batch_normalization_34_readvariableop_1_resource:	ђg
Xencoder_encoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђi
Zencoder_encoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђ]
Aencoder_encoder_block_23_conv2d_47_conv2d_readvariableop_resource:ђђQ
Bencoder_encoder_block_23_conv2d_47_biasadd_readvariableop_resource:	ђV
Gencoder_encoder_block_23_batch_normalization_35_readvariableop_resource:	ђX
Iencoder_encoder_block_23_batch_normalization_35_readvariableop_1_resource:	ђg
Xencoder_encoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђi
Zencoder_encoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђK
0encoder_conv2d_49_conv2d_readvariableop_resource:ђ?
1encoder_conv2d_49_biasadd_readvariableop_resource:C
/encoder_dense_12_matmul_readvariableop_resource:
ђђ?
0encoder_dense_12_biasadd_readvariableop_resource:	ђC
/encoder_dense_13_matmul_readvariableop_resource:
ђђ?
0encoder_dense_13_biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕб(encoder/conv2d_49/BiasAdd/ReadVariableOpб'encoder/conv2d_49/Conv2D/ReadVariableOpб'encoder/dense_12/BiasAdd/ReadVariableOpб&encoder/dense_12/MatMul/ReadVariableOpб'encoder/dense_13/BiasAdd/ReadVariableOpб&encoder/dense_13/MatMul/ReadVariableOpбOencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpбQencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б>encoder/encoder_block_20/batch_normalization_32/ReadVariableOpб@encoder/encoder_block_20/batch_normalization_32/ReadVariableOp_1б9encoder/encoder_block_20/conv2d_44/BiasAdd/ReadVariableOpб8encoder/encoder_block_20/conv2d_44/Conv2D/ReadVariableOpбOencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpбQencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б>encoder/encoder_block_21/batch_normalization_33/ReadVariableOpб@encoder/encoder_block_21/batch_normalization_33/ReadVariableOp_1б9encoder/encoder_block_21/conv2d_45/BiasAdd/ReadVariableOpб8encoder/encoder_block_21/conv2d_45/Conv2D/ReadVariableOpбOencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpбQencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б>encoder/encoder_block_22/batch_normalization_34/ReadVariableOpб@encoder/encoder_block_22/batch_normalization_34/ReadVariableOp_1б9encoder/encoder_block_22/conv2d_46/BiasAdd/ReadVariableOpб8encoder/encoder_block_22/conv2d_46/Conv2D/ReadVariableOpбOencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpбQencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б>encoder/encoder_block_23/batch_normalization_35/ReadVariableOpб@encoder/encoder_block_23/batch_normalization_35/ReadVariableOp_1б9encoder/encoder_block_23/conv2d_47/BiasAdd/ReadVariableOpб8encoder/encoder_block_23/conv2d_47/Conv2D/ReadVariableOp├
8encoder/encoder_block_20/conv2d_44/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_20_conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0с
)encoder/encoder_block_20/conv2d_44/Conv2DConv2Dinput_1@encoder/encoder_block_20/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
╣
9encoder/encoder_block_20/conv2d_44/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_20_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
*encoder/encoder_block_20/conv2d_44/BiasAddBiasAdd2encoder/encoder_block_20/conv2d_44/Conv2D:output:0Aencoder/encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђА
'encoder/encoder_block_20/conv2d_44/ReluRelu3encoder/encoder_block_20/conv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђр
1encoder/encoder_block_20/max_pooling2d_20/MaxPoolMaxPool5encoder/encoder_block_20/conv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_20/batch_normalization_32/ReadVariableOpReadVariableOpGencoder_encoder_block_20_batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0К
@encoder/encoder_block_20/batch_normalization_32/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_20_batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0т
Oencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
Qencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▀
@encoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_20/max_pooling2d_20/MaxPool:output:0Fencoder/encoder_block_20/batch_normalization_32/ReadVariableOp:value:0Hencoder/encoder_block_20/batch_normalization_32/ReadVariableOp_1:value:0Wencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ─
8encoder/encoder_block_21/conv2d_45/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_21_conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
)encoder/encoder_block_21/conv2d_45/Conv2DConv2DDencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3:y:0@encoder/encoder_block_21/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
╣
9encoder/encoder_block_21/conv2d_45/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_21_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0у
*encoder/encoder_block_21/conv2d_45/BiasAddBiasAdd2encoder/encoder_block_21/conv2d_45/Conv2D:output:0Aencoder/encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђЪ
'encoder/encoder_block_21/conv2d_45/ReluRelu3encoder/encoder_block_21/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђр
1encoder/encoder_block_21/max_pooling2d_21/MaxPoolMaxPool5encoder/encoder_block_21/conv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_21/batch_normalization_33/ReadVariableOpReadVariableOpGencoder_encoder_block_21_batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0К
@encoder/encoder_block_21/batch_normalization_33/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_21_batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0т
Oencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
Qencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▀
@encoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_21/max_pooling2d_21/MaxPool:output:0Fencoder/encoder_block_21/batch_normalization_33/ReadVariableOp:value:0Hencoder/encoder_block_21/batch_normalization_33/ReadVariableOp_1:value:0Wencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ─
8encoder/encoder_block_22/conv2d_46/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_22_conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
)encoder/encoder_block_22/conv2d_46/Conv2DConv2DDencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3:y:0@encoder/encoder_block_22/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
╣
9encoder/encoder_block_22/conv2d_46/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_22_conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0у
*encoder/encoder_block_22/conv2d_46/BiasAddBiasAdd2encoder/encoder_block_22/conv2d_46/Conv2D:output:0Aencoder/encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђЪ
'encoder/encoder_block_22/conv2d_46/ReluRelu3encoder/encoder_block_22/conv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђр
1encoder/encoder_block_22/max_pooling2d_22/MaxPoolMaxPool5encoder/encoder_block_22/conv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_22/batch_normalization_34/ReadVariableOpReadVariableOpGencoder_encoder_block_22_batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0К
@encoder/encoder_block_22/batch_normalization_34/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_22_batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0т
Oencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
Qencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▀
@encoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_22/max_pooling2d_22/MaxPool:output:0Fencoder/encoder_block_22/batch_normalization_34/ReadVariableOp:value:0Hencoder/encoder_block_22/batch_normalization_34/ReadVariableOp_1:value:0Wencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ─
8encoder/encoder_block_23/conv2d_47/Conv2D/ReadVariableOpReadVariableOpAencoder_encoder_block_23_conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0ъ
)encoder/encoder_block_23/conv2d_47/Conv2DConv2DDencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3:y:0@encoder/encoder_block_23/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
╣
9encoder/encoder_block_23/conv2d_47/BiasAdd/ReadVariableOpReadVariableOpBencoder_encoder_block_23_conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0у
*encoder/encoder_block_23/conv2d_47/BiasAddBiasAdd2encoder/encoder_block_23/conv2d_47/Conv2D:output:0Aencoder/encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЪ
'encoder/encoder_block_23/conv2d_47/ReluRelu3encoder/encoder_block_23/conv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђр
1encoder/encoder_block_23/max_pooling2d_23/MaxPoolMaxPool5encoder/encoder_block_23/conv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
├
>encoder/encoder_block_23/batch_normalization_35/ReadVariableOpReadVariableOpGencoder_encoder_block_23_batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0К
@encoder/encoder_block_23/batch_normalization_35/ReadVariableOp_1ReadVariableOpIencoder_encoder_block_23_batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0т
Oencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpXencoder_encoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ж
Qencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZencoder_encoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0▀
@encoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3:encoder/encoder_block_23/max_pooling2d_23/MaxPool:output:0Fencoder/encoder_block_23/batch_normalization_35/ReadVariableOp:value:0Hencoder/encoder_block_23/batch_normalization_35/ReadVariableOp_1:value:0Wencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Yencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( А
'encoder/conv2d_49/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0ч
encoder/conv2d_49/Conv2DConv2DDencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3:y:0/encoder/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
ќ
(encoder/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0│
encoder/conv2d_49/BiasAddBiasAdd!encoder/conv2d_49/Conv2D:output:00encoder/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         |
encoder/conv2d_49/ReluRelu"encoder/conv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:         h
encoder/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ъ
encoder/flatten_4/ReshapeReshape$encoder/conv2d_49/Relu:activations:0 encoder/flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђў
&encoder/dense_12/MatMul/ReadVariableOpReadVariableOp/encoder_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0е
encoder/dense_12/MatMulMatMul"encoder/flatten_4/Reshape:output:0.encoder/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
'encoder/dense_12/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ф
encoder/dense_12/BiasAddBiasAdd!encoder/dense_12/MatMul:product:0/encoder/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђў
&encoder/dense_13/MatMul/ReadVariableOpReadVariableOp/encoder_dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0е
encoder/dense_13/MatMulMatMul"encoder/flatten_4/Reshape:output:0.encoder/dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЋ
'encoder/dense_13/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ф
encoder/dense_13/BiasAddBiasAdd!encoder/dense_13/MatMul:product:0/encoder/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
encoder/sampling_4/ShapeShape!encoder/dense_12/BiasAdd:output:0*
T0*
_output_shapes
::ь¤p
&encoder/sampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(encoder/sampling_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(encoder/sampling_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 encoder/sampling_4/strided_sliceStridedSlice!encoder/sampling_4/Shape:output:0/encoder/sampling_4/strided_slice/stack:output:01encoder/sampling_4/strided_slice/stack_1:output:01encoder/sampling_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
encoder/sampling_4/Shape_1Shape!encoder/dense_12/BiasAdd:output:0*
T0*
_output_shapes
::ь¤r
(encoder/sampling_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*encoder/sampling_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"encoder/sampling_4/strided_slice_1StridedSlice#encoder/sampling_4/Shape_1:output:01encoder/sampling_4/strided_slice_1/stack:output:03encoder/sampling_4/strided_slice_1/stack_1:output:03encoder/sampling_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┤
&encoder/sampling_4/random_normal/shapePack)encoder/sampling_4/strided_slice:output:0+encoder/sampling_4/strided_slice_1:output:0*
N*
T0*
_output_shapes
:j
%encoder/sampling_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    l
'encoder/sampling_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?П
5encoder/sampling_4/random_normal/RandomStandardNormalRandomStandardNormal/encoder/sampling_4/random_normal/shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seed2ё Ў*
seed▒ т)л
$encoder/sampling_4/random_normal/mulMul>encoder/sampling_4/random_normal/RandomStandardNormal:output:00encoder/sampling_4/random_normal/stddev:output:0*
T0*(
_output_shapes
:         ђХ
 encoder/sampling_4/random_normalAddV2(encoder/sampling_4/random_normal/mul:z:0.encoder/sampling_4/random_normal/mean:output:0*
T0*(
_output_shapes
:         ђ]
encoder/sampling_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ќ
encoder/sampling_4/mulMul!encoder/sampling_4/mul/x:output:0!encoder/dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђl
encoder/sampling_4/ExpExpencoder/sampling_4/mul:z:0*
T0*(
_output_shapes
:         ђћ
encoder/sampling_4/mul_1Mulencoder/sampling_4/Exp:y:0$encoder/sampling_4/random_normal:z:0*
T0*(
_output_shapes
:         ђЊ
encoder/sampling_4/addAddV2!encoder/dense_12/BiasAdd:output:0encoder/sampling_4/mul_1:z:0*
T0*(
_output_shapes
:         ђq
IdentityIdentity!encoder/dense_12/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђs

Identity_1Identity!encoder/dense_13/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђl

Identity_2Identityencoder/sampling_4/add:z:0^NoOp*
T0*(
_output_shapes
:         ђ┼
NoOpNoOp)^encoder/conv2d_49/BiasAdd/ReadVariableOp(^encoder/conv2d_49/Conv2D/ReadVariableOp(^encoder/dense_12/BiasAdd/ReadVariableOp'^encoder/dense_12/MatMul/ReadVariableOp(^encoder/dense_13/BiasAdd/ReadVariableOp'^encoder/dense_13/MatMul/ReadVariableOpP^encoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_20/batch_normalization_32/ReadVariableOpA^encoder/encoder_block_20/batch_normalization_32/ReadVariableOp_1:^encoder/encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp9^encoder/encoder_block_20/conv2d_44/Conv2D/ReadVariableOpP^encoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_21/batch_normalization_33/ReadVariableOpA^encoder/encoder_block_21/batch_normalization_33/ReadVariableOp_1:^encoder/encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp9^encoder/encoder_block_21/conv2d_45/Conv2D/ReadVariableOpP^encoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_22/batch_normalization_34/ReadVariableOpA^encoder/encoder_block_22/batch_normalization_34/ReadVariableOp_1:^encoder/encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp9^encoder/encoder_block_22/conv2d_46/Conv2D/ReadVariableOpP^encoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpR^encoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?^encoder/encoder_block_23/batch_normalization_35/ReadVariableOpA^encoder/encoder_block_23/batch_normalization_35/ReadVariableOp_1:^encoder/encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp9^encoder/encoder_block_23/conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(encoder/conv2d_49/BiasAdd/ReadVariableOp(encoder/conv2d_49/BiasAdd/ReadVariableOp2R
'encoder/conv2d_49/Conv2D/ReadVariableOp'encoder/conv2d_49/Conv2D/ReadVariableOp2R
'encoder/dense_12/BiasAdd/ReadVariableOp'encoder/dense_12/BiasAdd/ReadVariableOp2P
&encoder/dense_12/MatMul/ReadVariableOp&encoder/dense_12/MatMul/ReadVariableOp2R
'encoder/dense_13/BiasAdd/ReadVariableOp'encoder/dense_13/BiasAdd/ReadVariableOp2P
&encoder/dense_13/MatMul/ReadVariableOp&encoder/dense_13/MatMul/ReadVariableOp2д
Qencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12б
Oencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2ё
@encoder/encoder_block_20/batch_normalization_32/ReadVariableOp_1@encoder/encoder_block_20/batch_normalization_32/ReadVariableOp_12ђ
>encoder/encoder_block_20/batch_normalization_32/ReadVariableOp>encoder/encoder_block_20/batch_normalization_32/ReadVariableOp2v
9encoder/encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp9encoder/encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_20/conv2d_44/Conv2D/ReadVariableOp8encoder/encoder_block_20/conv2d_44/Conv2D/ReadVariableOp2д
Qencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12б
Oencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2ё
@encoder/encoder_block_21/batch_normalization_33/ReadVariableOp_1@encoder/encoder_block_21/batch_normalization_33/ReadVariableOp_12ђ
>encoder/encoder_block_21/batch_normalization_33/ReadVariableOp>encoder/encoder_block_21/batch_normalization_33/ReadVariableOp2v
9encoder/encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp9encoder/encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_21/conv2d_45/Conv2D/ReadVariableOp8encoder/encoder_block_21/conv2d_45/Conv2D/ReadVariableOp2д
Qencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12б
Oencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2ё
@encoder/encoder_block_22/batch_normalization_34/ReadVariableOp_1@encoder/encoder_block_22/batch_normalization_34/ReadVariableOp_12ђ
>encoder/encoder_block_22/batch_normalization_34/ReadVariableOp>encoder/encoder_block_22/batch_normalization_34/ReadVariableOp2v
9encoder/encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp9encoder/encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_22/conv2d_46/Conv2D/ReadVariableOp8encoder/encoder_block_22/conv2d_46/Conv2D/ReadVariableOp2д
Qencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Qencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12б
Oencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpOencoder/encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2ё
@encoder/encoder_block_23/batch_normalization_35/ReadVariableOp_1@encoder/encoder_block_23/batch_normalization_35/ReadVariableOp_12ђ
>encoder/encoder_block_23/batch_normalization_35/ReadVariableOp>encoder/encoder_block_23/batch_normalization_35/ReadVariableOp2v
9encoder/encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp9encoder/encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp2t
8encoder/encoder_block_23/conv2d_47/Conv2D/ReadVariableOp8encoder/encoder_block_23/conv2d_47/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
я
б
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1910737

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
Ч
ъ
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1911044
input_tensorC
(conv2d_44_conv2d_readvariableop_resource:ђ8
)conv2d_44_biasadd_readvariableop_resource:	ђ=
.batch_normalization_32_readvariableop_resource:	ђ?
0batch_normalization_32_readvariableop_1_resource:	ђN
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб6batch_normalization_32/FusedBatchNormV3/ReadVariableOpб8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_32/ReadVariableOpб'batch_normalization_32/ReadVariableOp_1б conv2d_44/BiasAdd/ReadVariableOpбconv2d_44/Conv2D/ReadVariableOpЉ
conv2d_44/Conv2D/ReadVariableOpReadVariableOp(conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0Х
conv2d_44/Conv2DConv2Dinput_tensor'conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
Є
 conv2d_44/BiasAdd/ReadVariableOpReadVariableOp)conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ъ
conv2d_44/BiasAddBiasAddconv2d_44/Conv2D:output:0(conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђo
conv2d_44/ReluReluconv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђ»
max_pooling2d_20/MaxPoolMaxPoolconv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0╔
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_20/MaxPool:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Ѓ
IdentityIdentity+batch_normalization_32/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         @@ђЛ
NoOpNoOp7^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1!^conv2d_44/BiasAdd/ReadVariableOp ^conv2d_44/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2D
 conv2d_44/BiasAdd/ReadVariableOp conv2d_44/BiasAdd/ReadVariableOp2B
conv2d_44/Conv2D/ReadVariableOpconv2d_44/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nameinput_tensor
И
У
)__inference_encoder_layer_call_fn_1911340
input_1"
unknown:ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ
	unknown_7:	ђ
	unknown_8:	ђ
	unknown_9:	ђ

unknown_10:	ђ&

unknown_11:ђђ

unknown_12:	ђ

unknown_13:	ђ

unknown_14:	ђ

unknown_15:	ђ

unknown_16:	ђ&

unknown_17:ђђ

unknown_18:	ђ

unknown_19:	ђ

unknown_20:	ђ

unknown_21:	ђ

unknown_22:	ђ%

unknown_23:ђ

unknown_24:

unknown_25:
ђђ

unknown_26:	ђ

unknown_27:
ђђ

unknown_28:	ђ
identity

identity_1

identity_2ѕбStatefulPartitionedCallЂ
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
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ђ:         ђ:         ђ*8
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_1911273p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ђr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
а	
О
8__inference_batch_normalization_34_layer_call_fn_1912898

inputs
unknown:	ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1910643і
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
а'
ы
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1910874
input_tensorD
(conv2d_46_conv2d_readvariableop_resource:ђђ8
)conv2d_46_biasadd_readvariableop_resource:	ђ=
.batch_normalization_34_readvariableop_resource:	ђ?
0batch_normalization_34_readvariableop_1_resource:	ђN
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_34/AssignNewValueб'batch_normalization_34/AssignNewValue_1б6batch_normalization_34/FusedBatchNormV3/ReadVariableOpб8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_34/ReadVariableOpб'batch_normalization_34/ReadVariableOp_1б conv2d_46/BiasAdd/ReadVariableOpбconv2d_46/Conv2D/ReadVariableOpњ
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_46/Conv2DConv2Dinput_tensor'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Є
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђm
conv2d_46/ReluReluconv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђ»
max_pooling2d_22/MaxPoolMaxPoolconv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_22/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_34/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђБ
NoOpNoOp&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           ђ: : : : : : 2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:           ђ
&
_user_specified_nameinput_tensor
«┼
А!
D__inference_encoder_layer_call_and_return_conditional_losses_1912286
tensor_inputT
9encoder_block_20_conv2d_44_conv2d_readvariableop_resource:ђI
:encoder_block_20_conv2d_44_biasadd_readvariableop_resource:	ђN
?encoder_block_20_batch_normalization_32_readvariableop_resource:	ђP
Aencoder_block_20_batch_normalization_32_readvariableop_1_resource:	ђ_
Pencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:	ђU
9encoder_block_21_conv2d_45_conv2d_readvariableop_resource:ђђI
:encoder_block_21_conv2d_45_biasadd_readvariableop_resource:	ђN
?encoder_block_21_batch_normalization_33_readvariableop_resource:	ђP
Aencoder_block_21_batch_normalization_33_readvariableop_1_resource:	ђ_
Pencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђU
9encoder_block_22_conv2d_46_conv2d_readvariableop_resource:ђђI
:encoder_block_22_conv2d_46_biasadd_readvariableop_resource:	ђN
?encoder_block_22_batch_normalization_34_readvariableop_resource:	ђP
Aencoder_block_22_batch_normalization_34_readvariableop_1_resource:	ђ_
Pencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	ђU
9encoder_block_23_conv2d_47_conv2d_readvariableop_resource:ђђI
:encoder_block_23_conv2d_47_biasadd_readvariableop_resource:	ђN
?encoder_block_23_batch_normalization_35_readvariableop_resource:	ђP
Aencoder_block_23_batch_normalization_35_readvariableop_1_resource:	ђ_
Pencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђa
Rencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђC
(conv2d_49_conv2d_readvariableop_resource:ђ7
)conv2d_49_biasadd_readvariableop_resource:;
'dense_12_matmul_readvariableop_resource:
ђђ7
(dense_12_biasadd_readvariableop_resource:	ђ;
'dense_13_matmul_readvariableop_resource:
ђђ7
(dense_13_biasadd_readvariableop_resource:	ђ
identity

identity_1

identity_2ѕб conv2d_49/BiasAdd/ReadVariableOpбconv2d_49/Conv2D/ReadVariableOpбdense_12/BiasAdd/ReadVariableOpбdense_12/MatMul/ReadVariableOpбdense_13/BiasAdd/ReadVariableOpбdense_13/MatMul/ReadVariableOpбGencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpбIencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_20/batch_normalization_32/ReadVariableOpб8encoder_block_20/batch_normalization_32/ReadVariableOp_1б1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOpб0encoder_block_20/conv2d_44/Conv2D/ReadVariableOpбGencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpбIencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_21/batch_normalization_33/ReadVariableOpб8encoder_block_21/batch_normalization_33/ReadVariableOp_1б1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOpб0encoder_block_21/conv2d_45/Conv2D/ReadVariableOpбGencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpбIencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_22/batch_normalization_34/ReadVariableOpб8encoder_block_22/batch_normalization_34/ReadVariableOp_1б1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOpб0encoder_block_22/conv2d_46/Conv2D/ReadVariableOpбGencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpбIencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б6encoder_block_23/batch_normalization_35/ReadVariableOpб8encoder_block_23/batch_normalization_35/ReadVariableOp_1б1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOpб0encoder_block_23/conv2d_47/Conv2D/ReadVariableOp│
0encoder_block_20/conv2d_44/Conv2D/ReadVariableOpReadVariableOp9encoder_block_20_conv2d_44_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0п
!encoder_block_20/conv2d_44/Conv2DConv2Dtensor_input8encoder_block_20/conv2d_44/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђ*
paddingSAME*
strides
Е
1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_20_conv2d_44_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Л
"encoder_block_20/conv2d_44/BiasAddBiasAdd*encoder_block_20/conv2d_44/Conv2D:output:09encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:         ђђђЉ
encoder_block_20/conv2d_44/ReluRelu+encoder_block_20/conv2d_44/BiasAdd:output:0*
T0*2
_output_shapes 
:         ђђђЛ
)encoder_block_20/max_pooling2d_20/MaxPoolMaxPool-encoder_block_20/conv2d_44/Relu:activations:0*0
_output_shapes
:         @@ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_20/batch_normalization_32/ReadVariableOpReadVariableOp?encoder_block_20_batch_normalization_32_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_20/batch_normalization_32/ReadVariableOp_1ReadVariableOpAencoder_block_20_batch_normalization_32_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_20_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0»
8encoder_block_20/batch_normalization_32/FusedBatchNormV3FusedBatchNormV32encoder_block_20/max_pooling2d_20/MaxPool:output:0>encoder_block_20/batch_normalization_32/ReadVariableOp:value:0@encoder_block_20/batch_normalization_32/ReadVariableOp_1:value:0Oencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         @@ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ┤
0encoder_block_21/conv2d_45/Conv2D/ReadVariableOpReadVariableOp9encoder_block_21_conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0є
!encoder_block_21/conv2d_45/Conv2DConv2D<encoder_block_20/batch_normalization_32/FusedBatchNormV3:y:08encoder_block_21/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
Е
1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_21_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"encoder_block_21/conv2d_45/BiasAddBiasAdd*encoder_block_21/conv2d_45/Conv2D:output:09encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђЈ
encoder_block_21/conv2d_45/ReluRelu+encoder_block_21/conv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђЛ
)encoder_block_21/max_pooling2d_21/MaxPoolMaxPool-encoder_block_21/conv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_21/batch_normalization_33/ReadVariableOpReadVariableOp?encoder_block_21_batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_21/batch_normalization_33/ReadVariableOp_1ReadVariableOpAencoder_block_21_batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_21_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0»
8encoder_block_21/batch_normalization_33/FusedBatchNormV3FusedBatchNormV32encoder_block_21/max_pooling2d_21/MaxPool:output:0>encoder_block_21/batch_normalization_33/ReadVariableOp:value:0@encoder_block_21/batch_normalization_33/ReadVariableOp_1:value:0Oencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ┤
0encoder_block_22/conv2d_46/Conv2D/ReadVariableOpReadVariableOp9encoder_block_22_conv2d_46_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0є
!encoder_block_22/conv2d_46/Conv2DConv2D<encoder_block_21/batch_normalization_33/FusedBatchNormV3:y:08encoder_block_22/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђ*
paddingSAME*
strides
Е
1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_22_conv2d_46_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"encoder_block_22/conv2d_46/BiasAddBiasAdd*encoder_block_22/conv2d_46/Conv2D:output:09encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:           ђЈ
encoder_block_22/conv2d_46/ReluRelu+encoder_block_22/conv2d_46/BiasAdd:output:0*
T0*0
_output_shapes
:           ђЛ
)encoder_block_22/max_pooling2d_22/MaxPoolMaxPool-encoder_block_22/conv2d_46/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_22/batch_normalization_34/ReadVariableOpReadVariableOp?encoder_block_22_batch_normalization_34_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_22/batch_normalization_34/ReadVariableOp_1ReadVariableOpAencoder_block_22_batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_22_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0»
8encoder_block_22/batch_normalization_34/FusedBatchNormV3FusedBatchNormV32encoder_block_22/max_pooling2d_22/MaxPool:output:0>encoder_block_22/batch_normalization_34/ReadVariableOp:value:0@encoder_block_22/batch_normalization_34/ReadVariableOp_1:value:0Oencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ┤
0encoder_block_23/conv2d_47/Conv2D/ReadVariableOpReadVariableOp9encoder_block_23_conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0є
!encoder_block_23/conv2d_47/Conv2DConv2D<encoder_block_22/batch_normalization_34/FusedBatchNormV3:y:08encoder_block_23/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Е
1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp:encoder_block_23_conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0¤
"encoder_block_23/conv2d_47/BiasAddBiasAdd*encoder_block_23/conv2d_47/Conv2D:output:09encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЈ
encoder_block_23/conv2d_47/ReluRelu+encoder_block_23/conv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђЛ
)encoder_block_23/max_pooling2d_23/MaxPoolMaxPool-encoder_block_23/conv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
│
6encoder_block_23/batch_normalization_35/ReadVariableOpReadVariableOp?encoder_block_23_batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8encoder_block_23/batch_normalization_35/ReadVariableOp_1ReadVariableOpAencoder_block_23_batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Н
Gencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpPencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0┘
Iencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRencoder_block_23_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0»
8encoder_block_23/batch_normalization_35/FusedBatchNormV3FusedBatchNormV32encoder_block_23/max_pooling2d_23/MaxPool:output:0>encoder_block_23/batch_normalization_35/ReadVariableOp:value:0@encoder_block_23/batch_normalization_35/ReadVariableOp_1:value:0Oencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Qencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( Љ
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
:ђ*
dtype0с
conv2d_49/Conv2DConv2D<encoder_block_23/batch_normalization_35/FusedBatchNormV3:y:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingSAME*
strides
є
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Џ
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         l
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:         `
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Є
flatten_4/ReshapeReshapeconv2d_49/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:         ђѕ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0љ
dense_12/MatMulMatMulflatten_4/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЁ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0њ
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђѕ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype0љ
dense_13/MatMulMatMulflatten_4/Reshape:output:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЁ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0њ
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђg
sampling_4/ShapeShapedense_12/BiasAdd:output:0*
T0*
_output_shapes
::ь¤h
sampling_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 sampling_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 sampling_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
sampling_4/strided_sliceStridedSlicesampling_4/Shape:output:0'sampling_4/strided_slice/stack:output:0)sampling_4/strided_slice/stack_1:output:0)sampling_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
sampling_4/Shape_1Shapedense_12/BiasAdd:output:0*
T0*
_output_shapes
::ь¤j
 sampling_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:l
"sampling_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"sampling_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
sampling_4/strided_slice_1StridedSlicesampling_4/Shape_1:output:0)sampling_4/strided_slice_1/stack:output:0+sampling_4/strided_slice_1/stack_1:output:0+sampling_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskю
sampling_4/random_normal/shapePack!sampling_4/strided_slice:output:0#sampling_4/strided_slice_1:output:0*
N*
T0*
_output_shapes
:b
sampling_4/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
sampling_4/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╠
-sampling_4/random_normal/RandomStandardNormalRandomStandardNormal'sampling_4/random_normal/shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seed2»ЛH*
seed▒ т)И
sampling_4/random_normal/mulMul6sampling_4/random_normal/RandomStandardNormal:output:0(sampling_4/random_normal/stddev:output:0*
T0*(
_output_shapes
:         ђъ
sampling_4/random_normalAddV2 sampling_4/random_normal/mul:z:0&sampling_4/random_normal/mean:output:0*
T0*(
_output_shapes
:         ђU
sampling_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?~
sampling_4/mulMulsampling_4/mul/x:output:0dense_13/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ\
sampling_4/ExpExpsampling_4/mul:z:0*
T0*(
_output_shapes
:         ђ|
sampling_4/mul_1Mulsampling_4/Exp:y:0sampling_4/random_normal:z:0*
T0*(
_output_shapes
:         ђ{
sampling_4/addAddV2dense_12/BiasAdd:output:0sampling_4/mul_1:z:0*
T0*(
_output_shapes
:         ђi
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђk

Identity_1Identitydense_13/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         ђd

Identity_2Identitysampling_4/add:z:0^NoOp*
T0*(
_output_shapes
:         ђН
NoOpNoOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOpH^encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpJ^encoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_17^encoder_block_20/batch_normalization_32/ReadVariableOp9^encoder_block_20/batch_normalization_32/ReadVariableOp_12^encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp1^encoder_block_20/conv2d_44/Conv2D/ReadVariableOpH^encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpJ^encoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_17^encoder_block_21/batch_normalization_33/ReadVariableOp9^encoder_block_21/batch_normalization_33/ReadVariableOp_12^encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp1^encoder_block_21/conv2d_45/Conv2D/ReadVariableOpH^encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpJ^encoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_17^encoder_block_22/batch_normalization_34/ReadVariableOp9^encoder_block_22/batch_normalization_34/ReadVariableOp_12^encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp1^encoder_block_22/conv2d_46/Conv2D/ReadVariableOpH^encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpJ^encoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_17^encoder_block_23/batch_normalization_35/ReadVariableOp9^encoder_block_23/batch_normalization_35/ReadVariableOp_12^encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp1^encoder_block_23/conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2ќ
Iencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOpGencoder_block_20/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_20/batch_normalization_32/ReadVariableOp_18encoder_block_20/batch_normalization_32/ReadVariableOp_12p
6encoder_block_20/batch_normalization_32/ReadVariableOp6encoder_block_20/batch_normalization_32/ReadVariableOp2f
1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp1encoder_block_20/conv2d_44/BiasAdd/ReadVariableOp2d
0encoder_block_20/conv2d_44/Conv2D/ReadVariableOp0encoder_block_20/conv2d_44/Conv2D/ReadVariableOp2ќ
Iencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOpGencoder_block_21/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_21/batch_normalization_33/ReadVariableOp_18encoder_block_21/batch_normalization_33/ReadVariableOp_12p
6encoder_block_21/batch_normalization_33/ReadVariableOp6encoder_block_21/batch_normalization_33/ReadVariableOp2f
1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp1encoder_block_21/conv2d_45/BiasAdd/ReadVariableOp2d
0encoder_block_21/conv2d_45/Conv2D/ReadVariableOp0encoder_block_21/conv2d_45/Conv2D/ReadVariableOp2ќ
Iencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOpGencoder_block_22/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_22/batch_normalization_34/ReadVariableOp_18encoder_block_22/batch_normalization_34/ReadVariableOp_12p
6encoder_block_22/batch_normalization_34/ReadVariableOp6encoder_block_22/batch_normalization_34/ReadVariableOp2f
1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp1encoder_block_22/conv2d_46/BiasAdd/ReadVariableOp2d
0encoder_block_22/conv2d_46/Conv2D/ReadVariableOp0encoder_block_22/conv2d_46/Conv2D/ReadVariableOp2ќ
Iencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Iencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12њ
Gencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOpGencoder_block_23/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8encoder_block_23/batch_normalization_35/ReadVariableOp_18encoder_block_23/batch_normalization_35/ReadVariableOp_12p
6encoder_block_23/batch_normalization_35/ReadVariableOp6encoder_block_23/batch_normalization_35/ReadVariableOp2f
1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp1encoder_block_23/conv2d_47/BiasAdd/ReadVariableOp2d
0encoder_block_23/conv2d_47/Conv2D/ReadVariableOp0encoder_block_23/conv2d_47/Conv2D/ReadVariableOp:_ [
1
_output_shapes
:         ђђ
&
_user_specified_nametensor_input
а'
ы
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912604
input_tensorD
(conv2d_47_conv2d_readvariableop_resource:ђђ8
)conv2d_47_biasadd_readvariableop_resource:	ђ=
.batch_normalization_35_readvariableop_resource:	ђ?
0batch_normalization_35_readvariableop_1_resource:	ђN
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_35/AssignNewValueб'batch_normalization_35/AssignNewValue_1б6batch_normalization_35/FusedBatchNormV3/ReadVariableOpб8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_35/ReadVariableOpб'batch_normalization_35/ReadVariableOp_1б conv2d_47/BiasAdd/ReadVariableOpбconv2d_47/Conv2D/ReadVariableOpњ
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_47/Conv2DConv2Dinput_tensor'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
Є
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђm
conv2d_47/ReluReluconv2d_47/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ»
max_pooling2d_23/MaxPoolMaxPoolconv2d_47/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_23/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_35/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         ђБ
NoOpNoOp&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
а'
ы
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1910834
input_tensorD
(conv2d_45_conv2d_readvariableop_resource:ђђ8
)conv2d_45_biasadd_readvariableop_resource:	ђ=
.batch_normalization_33_readvariableop_resource:	ђ?
0batch_normalization_33_readvariableop_1_resource:	ђN
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:	ђP
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕб%batch_normalization_33/AssignNewValueб'batch_normalization_33/AssignNewValue_1б6batch_normalization_33/FusedBatchNormV3/ReadVariableOpб8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_33/ReadVariableOpб'batch_normalization_33/ReadVariableOp_1б conv2d_45/BiasAdd/ReadVariableOpбconv2d_45/Conv2D/ReadVariableOpњ
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype0┤
conv2d_45/Conv2DConv2Dinput_tensor'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђ*
paddingSAME*
strides
Є
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0ю
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         @@ђm
conv2d_45/ReluReluconv2d_45/BiasAdd:output:0*
T0*0
_output_shapes
:         @@ђ»
max_pooling2d_21/MaxPoolMaxPoolconv2d_45/Relu:activations:0*0
_output_shapes
:           ђ*
ksize
*
paddingVALID*
strides
Љ
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ћ
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0│
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0и
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0О
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_21/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(Ѓ
IdentityIdentity+batch_normalization_33/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:           ђБ
NoOpNoOp&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         @@ђ: : : : : : 2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         @@ђ
&
_user_specified_nameinput_tensor
╝	
ў
2__inference_encoder_block_23_layer_call_fn_1912561
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1910914x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         ђ
&
_user_specified_nameinput_tensor
ў
к
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1910491

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
╔Ь
ц"
 __inference__traced_save_1913224
file_prefix[
@read_disablecopyonread_encoder_encoder_block_20_conv2d_44_kernel:ђO
@read_1_disablecopyonread_encoder_encoder_block_20_conv2d_44_bias:	ђ]
Nread_2_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_gamma:	ђ\
Mread_3_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_beta:	ђc
Tread_4_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_moving_mean:	ђg
Xread_5_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_moving_variance:	ђ^
Bread_6_disablecopyonread_encoder_encoder_block_21_conv2d_45_kernel:ђђO
@read_7_disablecopyonread_encoder_encoder_block_21_conv2d_45_bias:	ђ]
Nread_8_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_gamma:	ђ\
Mread_9_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_beta:	ђd
Uread_10_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_moving_mean:	ђh
Yread_11_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_moving_variance:	ђ_
Cread_12_disablecopyonread_encoder_encoder_block_22_conv2d_46_kernel:ђђP
Aread_13_disablecopyonread_encoder_encoder_block_22_conv2d_46_bias:	ђ^
Oread_14_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_gamma:	ђ]
Nread_15_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_beta:	ђd
Uread_16_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_moving_mean:	ђh
Yread_17_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_moving_variance:	ђ_
Cread_18_disablecopyonread_encoder_encoder_block_23_conv2d_47_kernel:ђђP
Aread_19_disablecopyonread_encoder_encoder_block_23_conv2d_47_bias:	ђ^
Oread_20_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_gamma:	ђ]
Nread_21_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_beta:	ђd
Uread_22_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_moving_mean:	ђh
Yread_23_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_moving_variance:	ђM
2read_24_disablecopyonread_encoder_conv2d_49_kernel:ђ>
0read_25_disablecopyonread_encoder_conv2d_49_bias:E
1read_26_disablecopyonread_encoder_dense_12_kernel:
ђђ>
/read_27_disablecopyonread_encoder_dense_12_bias:	ђE
1read_28_disablecopyonread_encoder_dense_13_kernel:
ђђ>
/read_29_disablecopyonread_encoder_dense_13_bias:	ђ
savev2_const
identity_61ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_18/DisableCopyOnReadбRead_18/ReadVariableOpбRead_19/DisableCopyOnReadбRead_19/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_20/DisableCopyOnReadбRead_20/ReadVariableOpбRead_21/DisableCopyOnReadбRead_21/ReadVariableOpбRead_22/DisableCopyOnReadбRead_22/ReadVariableOpбRead_23/DisableCopyOnReadбRead_23/ReadVariableOpбRead_24/DisableCopyOnReadбRead_24/ReadVariableOpбRead_25/DisableCopyOnReadбRead_25/ReadVariableOpбRead_26/DisableCopyOnReadбRead_26/ReadVariableOpбRead_27/DisableCopyOnReadбRead_27/ReadVariableOpбRead_28/DisableCopyOnReadбRead_28/ReadVariableOpбRead_29/DisableCopyOnReadбRead_29/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
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
: њ
Read/DisableCopyOnReadDisableCopyOnRead@read_disablecopyonread_encoder_encoder_block_20_conv2d_44_kernel"/device:CPU:0*
_output_shapes
 ┼
Read/ReadVariableOpReadVariableOp@read_disablecopyonread_encoder_encoder_block_20_conv2d_44_kernel^Read/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:ђ*
dtype0r
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:ђj

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*'
_output_shapes
:ђћ
Read_1/DisableCopyOnReadDisableCopyOnRead@read_1_disablecopyonread_encoder_encoder_block_20_conv2d_44_bias"/device:CPU:0*
_output_shapes
 й
Read_1/ReadVariableOpReadVariableOp@read_1_disablecopyonread_encoder_encoder_block_20_conv2d_44_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђб
Read_2/DisableCopyOnReadDisableCopyOnReadNread_2_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_2/ReadVariableOpReadVariableOpNread_2_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђА
Read_3/DisableCopyOnReadDisableCopyOnReadMread_3_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_beta"/device:CPU:0*
_output_shapes
 ╩
Read_3/ReadVariableOpReadVariableOpMread_3_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђе
Read_4/DisableCopyOnReadDisableCopyOnReadTread_4_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_moving_mean"/device:CPU:0*
_output_shapes
 Л
Read_4/ReadVariableOpReadVariableOpTread_4_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђ`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђг
Read_5/DisableCopyOnReadDisableCopyOnReadXread_5_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_moving_variance"/device:CPU:0*
_output_shapes
 Н
Read_5/ReadVariableOpReadVariableOpXread_5_disablecopyonread_encoder_encoder_block_20_batch_normalization_32_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђќ
Read_6/DisableCopyOnReadDisableCopyOnReadBread_6_disablecopyonread_encoder_encoder_block_21_conv2d_45_kernel"/device:CPU:0*
_output_shapes
 ╠
Read_6/ReadVariableOpReadVariableOpBread_6_disablecopyonread_encoder_encoder_block_21_conv2d_45_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:ђђ*
dtype0x
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ђђo
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*(
_output_shapes
:ђђћ
Read_7/DisableCopyOnReadDisableCopyOnRead@read_7_disablecopyonread_encoder_encoder_block_21_conv2d_45_bias"/device:CPU:0*
_output_shapes
 й
Read_7/ReadVariableOpReadVariableOp@read_7_disablecopyonread_encoder_encoder_block_21_conv2d_45_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђб
Read_8/DisableCopyOnReadDisableCopyOnReadNread_8_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_gamma"/device:CPU:0*
_output_shapes
 ╦
Read_8/ReadVariableOpReadVariableOpNread_8_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђА
Read_9/DisableCopyOnReadDisableCopyOnReadMread_9_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_beta"/device:CPU:0*
_output_shapes
 ╩
Read_9/ReadVariableOpReadVariableOpMread_9_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђф
Read_10/DisableCopyOnReadDisableCopyOnReadUread_10_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_moving_mean"/device:CPU:0*
_output_shapes
 н
Read_10/ReadVariableOpReadVariableOpUread_10_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђ«
Read_11/DisableCopyOnReadDisableCopyOnReadYread_11_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_moving_variance"/device:CPU:0*
_output_shapes
 п
Read_11/ReadVariableOpReadVariableOpYread_11_disablecopyonread_encoder_encoder_block_21_batch_normalization_33_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђў
Read_12/DisableCopyOnReadDisableCopyOnReadCread_12_disablecopyonread_encoder_encoder_block_22_conv2d_46_kernel"/device:CPU:0*
_output_shapes
 ¤
Read_12/ReadVariableOpReadVariableOpCread_12_disablecopyonread_encoder_encoder_block_22_conv2d_46_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:ђђ*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ђђo
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:ђђќ
Read_13/DisableCopyOnReadDisableCopyOnReadAread_13_disablecopyonread_encoder_encoder_block_22_conv2d_46_bias"/device:CPU:0*
_output_shapes
 └
Read_13/ReadVariableOpReadVariableOpAread_13_disablecopyonread_encoder_encoder_block_22_conv2d_46_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђц
Read_14/DisableCopyOnReadDisableCopyOnReadOread_14_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_gamma"/device:CPU:0*
_output_shapes
 ╬
Read_14/ReadVariableOpReadVariableOpOread_14_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђБ
Read_15/DisableCopyOnReadDisableCopyOnReadNread_15_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_beta"/device:CPU:0*
_output_shapes
 ═
Read_15/ReadVariableOpReadVariableOpNread_15_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђф
Read_16/DisableCopyOnReadDisableCopyOnReadUread_16_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_moving_mean"/device:CPU:0*
_output_shapes
 н
Read_16/ReadVariableOpReadVariableOpUread_16_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђ«
Read_17/DisableCopyOnReadDisableCopyOnReadYread_17_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_moving_variance"/device:CPU:0*
_output_shapes
 п
Read_17/ReadVariableOpReadVariableOpYread_17_disablecopyonread_encoder_encoder_block_22_batch_normalization_34_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђў
Read_18/DisableCopyOnReadDisableCopyOnReadCread_18_disablecopyonread_encoder_encoder_block_23_conv2d_47_kernel"/device:CPU:0*
_output_shapes
 ¤
Read_18/ReadVariableOpReadVariableOpCread_18_disablecopyonread_encoder_encoder_block_23_conv2d_47_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:ђђ*
dtype0y
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:ђђo
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*(
_output_shapes
:ђђќ
Read_19/DisableCopyOnReadDisableCopyOnReadAread_19_disablecopyonread_encoder_encoder_block_23_conv2d_47_bias"/device:CPU:0*
_output_shapes
 └
Read_19/ReadVariableOpReadVariableOpAread_19_disablecopyonread_encoder_encoder_block_23_conv2d_47_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђц
Read_20/DisableCopyOnReadDisableCopyOnReadOread_20_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_gamma"/device:CPU:0*
_output_shapes
 ╬
Read_20/ReadVariableOpReadVariableOpOread_20_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђБ
Read_21/DisableCopyOnReadDisableCopyOnReadNread_21_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_beta"/device:CPU:0*
_output_shapes
 ═
Read_21/ReadVariableOpReadVariableOpNread_21_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђф
Read_22/DisableCopyOnReadDisableCopyOnReadUread_22_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_moving_mean"/device:CPU:0*
_output_shapes
 н
Read_22/ReadVariableOpReadVariableOpUread_22_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђ«
Read_23/DisableCopyOnReadDisableCopyOnReadYread_23_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_moving_variance"/device:CPU:0*
_output_shapes
 п
Read_23/ReadVariableOpReadVariableOpYread_23_disablecopyonread_encoder_encoder_block_23_batch_normalization_35_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЄ
Read_24/DisableCopyOnReadDisableCopyOnRead2read_24_disablecopyonread_encoder_conv2d_49_kernel"/device:CPU:0*
_output_shapes
 й
Read_24/ReadVariableOpReadVariableOp2read_24_disablecopyonread_encoder_conv2d_49_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:ђ*
dtype0x
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:ђn
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*'
_output_shapes
:ђЁ
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_encoder_conv2d_49_bias"/device:CPU:0*
_output_shapes
 «
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_encoder_conv2d_49_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:є
Read_26/DisableCopyOnReadDisableCopyOnRead1read_26_disablecopyonread_encoder_dense_12_kernel"/device:CPU:0*
_output_shapes
 х
Read_26/ReadVariableOpReadVariableOp1read_26_disablecopyonread_encoder_dense_12_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђg
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђё
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_encoder_dense_12_bias"/device:CPU:0*
_output_shapes
 «
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_encoder_dense_12_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђє
Read_28/DisableCopyOnReadDisableCopyOnRead1read_28_disablecopyonread_encoder_dense_13_kernel"/device:CPU:0*
_output_shapes
 х
Read_28/ReadVariableOpReadVariableOp1read_28_disablecopyonread_encoder_dense_13_kernel^Read_28/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ђђ*
dtype0q
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђg
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ђђё
Read_29/DisableCopyOnReadDisableCopyOnRead/read_29_disablecopyonread_encoder_dense_13_bias"/device:CPU:0*
_output_shapes
 «
Read_29/ReadVariableOpReadVariableOp/read_29_disablecopyonread_encoder_dense_13_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђ╬

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*э	
valueь	BЖ	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ё
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: щ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_61Identity_61:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ў
к
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913001

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
▒
v
G__inference_sampling_4_layer_call_and_return_conditional_losses_1912731
inputs_0
inputs_1
identityѕK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
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
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
Shape_1Shapeinputs_0*
T0*
_output_shapes
::ь¤_
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
 *  ђ?Х
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*
seed2ъы*
seed▒ т)Ќ
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*(
_output_shapes
:         ђ}
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*(
_output_shapes
:         ђJ
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?W
mulMulmul/x:output:0inputs_1*
T0*(
_output_shapes
:         ђF
ExpExpmul:z:0*
T0*(
_output_shapes
:         ђ[
mul_1MulExp:y:0random_normal:z:0*
T0*(
_output_shapes
:         ђT
addAddV2inputs_0	mul_1:z:0*
T0*(
_output_shapes
:         ђP
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:         ђ:         ђ:RN
(
_output_shapes
:         ђ
"
_user_specified_name
inputs_1:R N
(
_output_shapes
:         ђ
"
_user_specified_name
inputs_0
Й	
ў
2__inference_encoder_block_22_layer_call_fn_1912492
input_tensor#
unknown:ђђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:	ђ
	unknown_3:	ђ
	unknown_4:	ђ
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *V
fQRO
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1911122x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:           ђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:           ђ
&
_user_specified_nameinput_tensor
Ћ
i
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1912885

inputs
identityб
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
ў
к
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1910567

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
л
џ
*__inference_dense_13_layer_call_fn_1912689

inputs
unknown:
ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_1910979p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ў
к
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1910643

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0█
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
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
.:,                           ђн
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
я
б
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1910661

inputs&
readvariableop_resource:	ђ(
readvariableop_1_resource:	ђ7
(fusedbatchnormv3_readvariableop_resource:	ђ9
*fusedbatchnormv3_readvariableop_1_resource:	ђ
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ђ*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ђ*
dtype0Ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ђ*
dtype0═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
epsilon%oЃ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           ђ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           ђ: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:j f
B
_output_shapes0
.:,                           ђ
 
_user_specified_nameinputs
Ћ
i
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1910618

inputs
identityб
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
ў
к
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1910719

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
 
_user_specified_nameinputs"з
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┤
serving_defaultа
E
input_1:
serving_default_input_1:0         ђђ=
output_11
StatefulPartitionedCall:0         ђ=
output_21
StatefulPartitionedCall:1         ђ=
output_31
StatefulPartitionedCall:2         ђtensorflow/serving/predict:┤ш
о
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

block5
	final_cov

flattening

z_mean
z_logvar
	embedding

signatures"
_tf_keras_model
є
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17
%18
&19
'20
(21
)22
*23
+24
,25
-26
.27
/28
029"
trackable_list_wrapper
к
0
1
2
3
4
5
6
7
8
 9
!10
"11
%12
&13
'14
(15
+16
,17
-18
.19
/20
021"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Н
6trace_0
7trace_1
8trace_2
9trace_32Ж
)__inference_encoder_layer_call_fn_1911340
)__inference_encoder_layer_call_fn_1911484
)__inference_encoder_layer_call_fn_1911945
)__inference_encoder_layer_call_fn_1912014╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
 z6trace_0z7trace_1z8trace_2z9trace_3
┴
:trace_0
;trace_1
<trace_2
=trace_32о
D__inference_encoder_layer_call_and_return_conditional_losses_1911016
D__inference_encoder_layer_call_and_return_conditional_losses_1911195
D__inference_encoder_layer_call_and_return_conditional_losses_1912150
D__inference_encoder_layer_call_and_return_conditional_losses_1912286╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
 z:trace_0z;trace_1z<trace_2z=trace_3
═B╩
"__inference__wrapped_model_1910460input_1"ў
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
─
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses
Dconv
Epooling
Fbn"
_tf_keras_layer
─
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Mconv
Npooling
Obn"
_tf_keras_layer
─
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
Vconv
Wpooling
Xbn"
_tf_keras_layer
─
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses
_conv
`pooling
abn"
_tf_keras_layer
G
b	keras_api
cconv
dpooling
ebn"
_tf_keras_layer
П
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

+kernel
,bias
 l_jit_compiled_convolution_op"
_tf_keras_layer
Ц
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
╗
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
ф
	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses"
_tf_keras_layer
-
Ёserving_default"
signature_map
D:Bђ2)encoder/encoder_block_20/conv2d_44/kernel
6:4ђ2'encoder/encoder_block_20/conv2d_44/bias
D:Bђ25encoder/encoder_block_20/batch_normalization_32/gamma
C:Aђ24encoder/encoder_block_20/batch_normalization_32/beta
L:Jђ (2;encoder/encoder_block_20/batch_normalization_32/moving_mean
P:Nђ (2?encoder/encoder_block_20/batch_normalization_32/moving_variance
E:Cђђ2)encoder/encoder_block_21/conv2d_45/kernel
6:4ђ2'encoder/encoder_block_21/conv2d_45/bias
D:Bђ25encoder/encoder_block_21/batch_normalization_33/gamma
C:Aђ24encoder/encoder_block_21/batch_normalization_33/beta
L:Jђ (2;encoder/encoder_block_21/batch_normalization_33/moving_mean
P:Nђ (2?encoder/encoder_block_21/batch_normalization_33/moving_variance
E:Cђђ2)encoder/encoder_block_22/conv2d_46/kernel
6:4ђ2'encoder/encoder_block_22/conv2d_46/bias
D:Bђ25encoder/encoder_block_22/batch_normalization_34/gamma
C:Aђ24encoder/encoder_block_22/batch_normalization_34/beta
L:Jђ (2;encoder/encoder_block_22/batch_normalization_34/moving_mean
P:Nђ (2?encoder/encoder_block_22/batch_normalization_34/moving_variance
E:Cђђ2)encoder/encoder_block_23/conv2d_47/kernel
6:4ђ2'encoder/encoder_block_23/conv2d_47/bias
D:Bђ25encoder/encoder_block_23/batch_normalization_35/gamma
C:Aђ24encoder/encoder_block_23/batch_normalization_35/beta
L:Jђ (2;encoder/encoder_block_23/batch_normalization_35/moving_mean
P:Nђ (2?encoder/encoder_block_23/batch_normalization_35/moving_variance
3:1ђ2encoder/conv2d_49/kernel
$:"2encoder/conv2d_49/bias
+:)
ђђ2encoder/dense_12/kernel
$:"ђ2encoder/dense_12/bias
+:)
ђђ2encoder/dense_13/kernel
$:"ђ2encoder/dense_13/bias
X
0
1
2
3
#4
$5
)6
*7"
trackable_list_wrapper
f
0
	1

2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBЗ
)__inference_encoder_layer_call_fn_1911340input_1"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
эBЗ
)__inference_encoder_layer_call_fn_1911484input_1"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
ЧBщ
)__inference_encoder_layer_call_fn_1911945tensor_input"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
ЧBщ
)__inference_encoder_layer_call_fn_1912014tensor_input"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
њBЈ
D__inference_encoder_layer_call_and_return_conditional_losses_1911016input_1"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
њBЈ
D__inference_encoder_layer_call_and_return_conditional_losses_1911195input_1"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
ЌBћ
D__inference_encoder_layer_call_and_return_conditional_losses_1912150tensor_input"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
ЌBћ
D__inference_encoder_layer_call_and_return_conditional_losses_1912286tensor_input"╗
┤▓░
FullArgSpec
argsџ
jtensor_input
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
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
Н
Іtrace_0
їtrace_12џ
2__inference_encoder_block_20_layer_call_fn_1912303
2__inference_encoder_block_20_layer_call_fn_1912320»
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
І
Їtrace_0
јtrace_12л
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912346
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912372»
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
С
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses

kernel
bias
!Ћ_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ќ	variables
Ќtrainable_variables
ўregularization_losses
Ў	keras_api
џ__call__
+Џ&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses
	бaxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Бnon_trainable_variables
цlayers
Цmetrics
 дlayer_regularization_losses
Дlayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
Н
еtrace_0
Еtrace_12џ
2__inference_encoder_block_21_layer_call_fn_1912389
2__inference_encoder_block_21_layer_call_fn_1912406»
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
 zеtrace_0zЕtrace_1
І
фtrace_0
Фtrace_12л
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912432
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912458»
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
С
г	variables
Гtrainable_variables
«regularization_losses
»	keras_api
░__call__
+▒&call_and_return_all_conditional_losses

kernel
bias
!▓_jit_compiled_convolution_op"
_tf_keras_layer
Ф
│	variables
┤trainable_variables
хregularization_losses
Х	keras_api
и__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
╣	variables
║trainable_variables
╗regularization_losses
╝	keras_api
й__call__
+Й&call_and_return_all_conditional_losses
	┐axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
 1
!2
"3
#4
$5"
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Н
┼trace_0
кtrace_12џ
2__inference_encoder_block_22_layer_call_fn_1912475
2__inference_encoder_block_22_layer_call_fn_1912492»
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
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912518
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912544»
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

kernel
 bias
!¤_jit_compiled_convolution_op"
_tf_keras_layer
Ф
л	variables
Лtrainable_variables
мregularization_losses
М	keras_api
н__call__
+Н&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
о	variables
Оtrainable_variables
пregularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses
	▄axis
	!gamma
"beta
#moving_mean
$moving_variance"
_tf_keras_layer
J
%0
&1
'2
(3
)4
*5"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Пnon_trainable_variables
яlayers
▀metrics
 Яlayer_regularization_losses
рlayer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
Н
Рtrace_0
сtrace_12џ
2__inference_encoder_block_23_layer_call_fn_1912561
2__inference_encoder_block_23_layer_call_fn_1912578»
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
 zРtrace_0zсtrace_1
І
Сtrace_0
тtrace_12л
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912604
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912630»
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
 zСtrace_0zтtrace_1
С
Т	variables
уtrainable_variables
Уregularization_losses
ж	keras_api
Ж__call__
+в&call_and_return_all_conditional_losses

%kernel
&bias
!В_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ь	variables
Ьtrainable_variables
№regularization_losses
­	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
з	variables
Зtrainable_variables
шregularization_losses
Ш	keras_api
э__call__
+Э&call_and_return_all_conditional_losses
	щaxis
	'gamma
(beta
)moving_mean
*moving_variance"
_tf_keras_layer
"
_generic_user_object
L
Щ	keras_api
!ч_jit_compiled_convolution_op"
_tf_keras_layer
)
Ч	keras_api"
_tf_keras_layer
)
§	keras_api"
_tf_keras_layer
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
■non_trainable_variables
 layers
ђmetrics
 Ђlayer_regularization_losses
ѓlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
у
Ѓtrace_02╚
+__inference_conv2d_49_layer_call_fn_1912639ў
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
 zЃtrace_0
ѓ
ёtrace_02с
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1912650ў
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
▓
Ёnon_trainable_variables
єlayers
Єmetrics
 ѕlayer_regularization_losses
Ѕlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
у
іtrace_02╚
+__inference_flatten_4_layer_call_fn_1912655ў
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
 zіtrace_0
ѓ
Іtrace_02с
F__inference_flatten_4_layer_call_and_return_conditional_losses_1912661ў
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
 zІtrace_0
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
Т
Љtrace_02К
*__inference_dense_12_layer_call_fn_1912670ў
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
 zЉtrace_0
Ђ
њtrace_02Р
E__inference_dense_12_layer_call_and_return_conditional_losses_1912680ў
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
 zњtrace_0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
Т
ўtrace_02К
*__inference_dense_13_layer_call_fn_1912689ў
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
 zўtrace_0
Ђ
Ўtrace_02Р
E__inference_dense_13_layer_call_and_return_conditional_losses_1912699ў
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
 zЎtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и
џnon_trainable_variables
Џlayers
юmetrics
 Юlayer_regularization_losses
ъlayer_metrics
	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
У
Ъtrace_02╔
,__inference_sampling_4_layer_call_fn_1912705ў
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
 zЪtrace_0
Ѓ
аtrace_02С
G__inference_sampling_4_layer_call_and_return_conditional_losses_1912731ў
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
 zаtrace_0
╠B╔
%__inference_signature_wrapper_1911876input_1"ћ
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
.
0
1"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
2__inference_encoder_block_20_layer_call_fn_1912303input_tensor"»
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
2__inference_encoder_block_20_layer_call_fn_1912320input_tensor"»
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
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912346input_tensor"»
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
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912372input_tensor"»
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Аnon_trainable_variables
бlayers
Бmetrics
 цlayer_regularization_losses
Цlayer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
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
дnon_trainable_variables
Дlayers
еmetrics
 Еlayer_regularization_losses
фlayer_metrics
ќ	variables
Ќtrainable_variables
ўregularization_losses
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
Ь
Фtrace_02¤
2__inference_max_pooling2d_20_layer_call_fn_1912736ў
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
 zФtrace_0
Ѕ
гtrace_02Ж
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1912741ў
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
 zгtrace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Гnon_trainable_variables
«layers
»metrics
 ░layer_regularization_losses
▒layer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
у
▓trace_0
│trace_12г
8__inference_batch_normalization_32_layer_call_fn_1912754
8__inference_batch_normalization_32_layer_call_fn_1912767х
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
 z▓trace_0z│trace_1
Ю
┤trace_0
хtrace_12Р
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912785
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912803х
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
 z┤trace_0zхtrace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
M0
N1
O2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
2__inference_encoder_block_21_layer_call_fn_1912389input_tensor"»
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
2__inference_encoder_block_21_layer_call_fn_1912406input_tensor"»
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
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912432input_tensor"»
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
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912458input_tensor"»
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
г	variables
Гtrainable_variables
«regularization_losses
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
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
╗non_trainable_variables
╝layers
йmetrics
 Йlayer_regularization_losses
┐layer_metrics
│	variables
┤trainable_variables
хregularization_losses
и__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
Ь
└trace_02¤
2__inference_max_pooling2d_21_layer_call_fn_1912808ў
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
 z└trace_0
Ѕ
┴trace_02Ж
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1912813ў
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
 z┴trace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
кlayer_metrics
╣	variables
║trainable_variables
╗regularization_losses
й__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
у
Кtrace_0
╚trace_12г
8__inference_batch_normalization_33_layer_call_fn_1912826
8__inference_batch_normalization_33_layer_call_fn_1912839х
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
 zКtrace_0z╚trace_1
Ю
╔trace_0
╩trace_12Р
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912857
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912875х
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
 z╔trace_0z╩trace_1
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
2__inference_encoder_block_22_layer_call_fn_1912475input_tensor"»
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
2__inference_encoder_block_22_layer_call_fn_1912492input_tensor"»
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
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912518input_tensor"»
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
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912544input_tensor"»
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
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
лnon_trainable_variables
Лlayers
мmetrics
 Мlayer_regularization_losses
нlayer_metrics
л	variables
Лtrainable_variables
мregularization_losses
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
Ь
Нtrace_02¤
2__inference_max_pooling2d_22_layer_call_fn_1912880ў
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
 zНtrace_0
Ѕ
оtrace_02Ж
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1912885ў
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
 zоtrace_0
<
!0
"1
#2
$3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Оnon_trainable_variables
пlayers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
о	variables
Оtrainable_variables
пregularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
у
▄trace_0
Пtrace_12г
8__inference_batch_normalization_34_layer_call_fn_1912898
8__inference_batch_normalization_34_layer_call_fn_1912911х
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
 z▄trace_0zПtrace_1
Ю
яtrace_0
▀trace_12Р
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912929
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912947х
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
 zяtrace_0z▀trace_1
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
5
_0
`1
a2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBШ
2__inference_encoder_block_23_layer_call_fn_1912561input_tensor"»
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
2__inference_encoder_block_23_layer_call_fn_1912578input_tensor"»
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
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912604input_tensor"»
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
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912630input_tensor"»
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
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Яnon_trainable_variables
рlayers
Рmetrics
 сlayer_regularization_losses
Сlayer_metrics
Т	variables
уtrainable_variables
Уregularization_losses
Ж__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
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
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
ь	variables
Ьtrainable_variables
№regularization_losses
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
Ь
Жtrace_02¤
2__inference_max_pooling2d_23_layer_call_fn_1912952ў
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
 zЖtrace_0
Ѕ
вtrace_02Ж
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1912957ў
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
 zвtrace_0
<
'0
(1
)2
*3"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вnon_trainable_variables
ьlayers
Ьmetrics
 №layer_regularization_losses
­layer_metrics
з	variables
Зtrainable_variables
шregularization_losses
э__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
у
ыtrace_0
Ыtrace_12г
8__inference_batch_normalization_35_layer_call_fn_1912970
8__inference_batch_normalization_35_layer_call_fn_1912983х
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
 zыtrace_0zЫtrace_1
Ю
зtrace_0
Зtrace_12Р
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913001
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913019х
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
 zзtrace_0zЗtrace_1
 "
trackable_list_wrapper
"
_generic_user_object
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
_generic_user_object
"
_generic_user_object
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
+__inference_conv2d_49_layer_call_fn_1912639inputs"ў
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
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1912650inputs"ў
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
+__inference_flatten_4_layer_call_fn_1912655inputs"ў
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
F__inference_flatten_4_layer_call_and_return_conditional_losses_1912661inputs"ў
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
нBЛ
*__inference_dense_12_layer_call_fn_1912670inputs"ў
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
E__inference_dense_12_layer_call_and_return_conditional_losses_1912680inputs"ў
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
нBЛ
*__inference_dense_13_layer_call_fn_1912689inputs"ў
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
E__inference_dense_13_layer_call_and_return_conditional_losses_1912699inputs"ў
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
РB▀
,__inference_sampling_4_layer_call_fn_1912705inputs_0inputs_1"ў
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
§BЩ
G__inference_sampling_4_layer_call_and_return_conditional_losses_1912731inputs_0inputs_1"ў
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
▄B┘
2__inference_max_pooling2d_20_layer_call_fn_1912736inputs"ў
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
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1912741inputs"ў
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
0
1"
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
8__inference_batch_normalization_32_layer_call_fn_1912754inputs"х
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
8__inference_batch_normalization_32_layer_call_fn_1912767inputs"х
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912785inputs"х
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
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912803inputs"х
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
▄B┘
2__inference_max_pooling2d_21_layer_call_fn_1912808inputs"ў
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
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1912813inputs"ў
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
0
1"
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
8__inference_batch_normalization_33_layer_call_fn_1912826inputs"х
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
8__inference_batch_normalization_33_layer_call_fn_1912839inputs"х
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912857inputs"х
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
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912875inputs"х
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
▄B┘
2__inference_max_pooling2d_22_layer_call_fn_1912880inputs"ў
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
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1912885inputs"ў
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
#0
$1"
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
8__inference_batch_normalization_34_layer_call_fn_1912898inputs"х
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
8__inference_batch_normalization_34_layer_call_fn_1912911inputs"х
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912929inputs"х
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
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912947inputs"х
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
▄B┘
2__inference_max_pooling2d_23_layer_call_fn_1912952inputs"ў
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
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1912957inputs"ў
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
)0
*1"
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
8__inference_batch_normalization_35_layer_call_fn_1912970inputs"х
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
8__inference_batch_normalization_35_layer_call_fn_1912983inputs"х
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913001inputs"х
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913019inputs"х
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
 Ю
"__inference__wrapped_model_1910460Ш !"#$%&'()*+,-./0:б7
0б-
+і(
input_1         ђђ
ф "ЌфЊ
/
output_1#і 
output_1         ђ
/
output_2#і 
output_2         ђ
/
output_3#і 
output_3         ђч
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912785БRбO
HбE
;і8
inputs,                           ђ
p

 
ф "GбD
=і:
tensor_0,                           ђ
џ ч
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_1912803БRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "GбD
=і:
tensor_0,                           ђ
џ Н
8__inference_batch_normalization_32_layer_call_fn_1912754ўRбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђН
8__inference_batch_normalization_32_layer_call_fn_1912767ўRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђч
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912857БRбO
HбE
;і8
inputs,                           ђ
p

 
ф "GбD
=і:
tensor_0,                           ђ
џ ч
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_1912875БRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "GбD
=і:
tensor_0,                           ђ
џ Н
8__inference_batch_normalization_33_layer_call_fn_1912826ўRбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђН
8__inference_batch_normalization_33_layer_call_fn_1912839ўRбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђч
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912929Б!"#$RбO
HбE
;і8
inputs,                           ђ
p

 
ф "GбD
=і:
tensor_0,                           ђ
џ ч
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_1912947Б!"#$RбO
HбE
;і8
inputs,                           ђ
p 

 
ф "GбD
=і:
tensor_0,                           ђ
џ Н
8__inference_batch_normalization_34_layer_call_fn_1912898ў!"#$RбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђН
8__inference_batch_normalization_34_layer_call_fn_1912911ў!"#$RбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђч
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913001Б'()*RбO
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
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_1913019Б'()*RбO
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
8__inference_batch_normalization_35_layer_call_fn_1912970ў'()*RбO
HбE
;і8
inputs,                           ђ
p

 
ф "<і9
unknown,                           ђН
8__inference_batch_normalization_35_layer_call_fn_1912983ў'()*RбO
HбE
;і8
inputs,                           ђ
p 

 
ф "<і9
unknown,                           ђЙ
F__inference_conv2d_49_layer_call_and_return_conditional_losses_1912650t+,8б5
.б+
)і&
inputs         ђ
ф "4б1
*і'
tensor_0         
џ ў
+__inference_conv2d_49_layer_call_fn_1912639i+,8б5
.б+
)і&
inputs         ђ
ф ")і&
unknown         «
E__inference_dense_12_layer_call_and_return_conditional_losses_1912680e-.0б-
&б#
!і
inputs         ђ
ф "-б*
#і 
tensor_0         ђ
џ ѕ
*__inference_dense_12_layer_call_fn_1912670Z-.0б-
&б#
!і
inputs         ђ
ф ""і
unknown         ђ«
E__inference_dense_13_layer_call_and_return_conditional_losses_1912699e/00б-
&б#
!і
inputs         ђ
ф "-б*
#і 
tensor_0         ђ
џ ѕ
*__inference_dense_13_layer_call_fn_1912689Z/00б-
&б#
!і
inputs         ђ
ф ""і
unknown         ђо
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912346ёCб@
9б6
0і-
input_tensor         ђђ
p
ф "5б2
+і(
tensor_0         @@ђ
џ о
M__inference_encoder_block_20_layer_call_and_return_conditional_losses_1912372ёCб@
9б6
0і-
input_tensor         ђђ
p 
ф "5б2
+і(
tensor_0         @@ђ
џ »
2__inference_encoder_block_20_layer_call_fn_1912303yCб@
9б6
0і-
input_tensor         ђђ
p
ф "*і'
unknown         @@ђ»
2__inference_encoder_block_20_layer_call_fn_1912320yCб@
9б6
0і-
input_tensor         ђђ
p 
ф "*і'
unknown         @@ђН
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912432ЃBб?
8б5
/і,
input_tensor         @@ђ
p
ф "5б2
+і(
tensor_0           ђ
џ Н
M__inference_encoder_block_21_layer_call_and_return_conditional_losses_1912458ЃBб?
8б5
/і,
input_tensor         @@ђ
p 
ф "5б2
+і(
tensor_0           ђ
џ «
2__inference_encoder_block_21_layer_call_fn_1912389xBб?
8б5
/і,
input_tensor         @@ђ
p
ф "*і'
unknown           ђ«
2__inference_encoder_block_21_layer_call_fn_1912406xBб?
8б5
/і,
input_tensor         @@ђ
p 
ф "*і'
unknown           ђН
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912518Ѓ !"#$Bб?
8б5
/і,
input_tensor           ђ
p
ф "5б2
+і(
tensor_0         ђ
џ Н
M__inference_encoder_block_22_layer_call_and_return_conditional_losses_1912544Ѓ !"#$Bб?
8б5
/і,
input_tensor           ђ
p 
ф "5б2
+і(
tensor_0         ђ
џ «
2__inference_encoder_block_22_layer_call_fn_1912475x !"#$Bб?
8б5
/і,
input_tensor           ђ
p
ф "*і'
unknown         ђ«
2__inference_encoder_block_22_layer_call_fn_1912492x !"#$Bб?
8б5
/і,
input_tensor           ђ
p 
ф "*і'
unknown         ђН
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912604Ѓ%&'()*Bб?
8б5
/і,
input_tensor         ђ
p
ф "5б2
+і(
tensor_0         ђ
џ Н
M__inference_encoder_block_23_layer_call_and_return_conditional_losses_1912630Ѓ%&'()*Bб?
8б5
/і,
input_tensor         ђ
p 
ф "5б2
+і(
tensor_0         ђ
џ «
2__inference_encoder_block_23_layer_call_fn_1912561x%&'()*Bб?
8б5
/і,
input_tensor         ђ
p
ф "*і'
unknown         ђ«
2__inference_encoder_block_23_layer_call_fn_1912578x%&'()*Bб?
8б5
/і,
input_tensor         ђ
p 
ф "*і'
unknown         ђ║
D__inference_encoder_layer_call_and_return_conditional_losses_1911016ы !"#$%&'()*+,-./0JбG
0б-
+і(
input_1         ђђ
ф

trainingp"ѓб
xџu
%і"

tensor_0_0         ђ
%і"

tensor_0_1         ђ
%і"

tensor_0_2         ђ
џ ║
D__inference_encoder_layer_call_and_return_conditional_losses_1911195ы !"#$%&'()*+,-./0JбG
0б-
+і(
input_1         ђђ
ф

trainingp "ѓб
xџu
%і"

tensor_0_0         ђ
%і"

tensor_0_1         ђ
%і"

tensor_0_2         ђ
џ ┐
D__inference_encoder_layer_call_and_return_conditional_losses_1912150Ш !"#$%&'()*+,-./0OбL
5б2
0і-
tensor_input         ђђ
ф

trainingp"ѓб
xџu
%і"

tensor_0_0         ђ
%і"

tensor_0_1         ђ
%і"

tensor_0_2         ђ
џ ┐
D__inference_encoder_layer_call_and_return_conditional_losses_1912286Ш !"#$%&'()*+,-./0OбL
5б2
0і-
tensor_input         ђђ
ф

trainingp "ѓб
xџu
%і"

tensor_0_0         ђ
%і"

tensor_0_1         ђ
%і"

tensor_0_2         ђ
џ ј
)__inference_encoder_layer_call_fn_1911340Я !"#$%&'()*+,-./0JбG
0б-
+і(
input_1         ђђ
ф

trainingp"rџo
#і 
tensor_0         ђ
#і 
tensor_1         ђ
#і 
tensor_2         ђј
)__inference_encoder_layer_call_fn_1911484Я !"#$%&'()*+,-./0JбG
0б-
+і(
input_1         ђђ
ф

trainingp "rџo
#і 
tensor_0         ђ
#і 
tensor_1         ђ
#і 
tensor_2         ђЊ
)__inference_encoder_layer_call_fn_1911945т !"#$%&'()*+,-./0OбL
5б2
0і-
tensor_input         ђђ
ф

trainingp"rџo
#і 
tensor_0         ђ
#і 
tensor_1         ђ
#і 
tensor_2         ђЊ
)__inference_encoder_layer_call_fn_1912014т !"#$%&'()*+,-./0OбL
5б2
0і-
tensor_input         ђђ
ф

trainingp "rџo
#і 
tensor_0         ђ
#і 
tensor_1         ђ
#і 
tensor_2         ђ▓
F__inference_flatten_4_layer_call_and_return_conditional_losses_1912661h7б4
-б*
(і%
inputs         
ф "-б*
#і 
tensor_0         ђ
џ ї
+__inference_flatten_4_layer_call_fn_1912655]7б4
-б*
(і%
inputs         
ф ""і
unknown         ђэ
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_1912741ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_max_pooling2d_20_layer_call_fn_1912736џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    э
M__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_1912813ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_max_pooling2d_21_layer_call_fn_1912808џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    э
M__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_1912885ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_max_pooling2d_22_layer_call_fn_1912880џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    э
M__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_1912957ЦRбO
HбE
Cі@
inputs4                                    
ф "OбL
EіB
tensor_04                                    
џ Л
2__inference_max_pooling2d_23_layer_call_fn_1912952џRбO
HбE
Cі@
inputs4                                    
ф "DіA
unknown4                                    ┘
G__inference_sampling_4_layer_call_and_return_conditional_losses_1912731Ї\бY
RбO
MџJ
#і 
inputs_0         ђ
#і 
inputs_1         ђ
ф "-б*
#і 
tensor_0         ђ
џ │
,__inference_sampling_4_layer_call_fn_1912705ѓ\бY
RбO
MџJ
#і 
inputs_0         ђ
#і 
inputs_1         ђ
ф ""і
unknown         ђФ
%__inference_signature_wrapper_1911876Ђ !"#$%&'()*+,-./0EбB
б 
;ф8
6
input_1+і(
input_1         ђђ"ЌфЊ
/
output_1#і 
output_1         ђ
/
output_2#і 
output_2         ђ
/
output_3#і 
output_3         ђ