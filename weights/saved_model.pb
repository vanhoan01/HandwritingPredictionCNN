��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
.
Identity

input"T
output"T"	
Ttype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
executor_typestring �
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
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
�
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	� *
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	� *
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
: *
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

: 
*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
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

NoOpNoOp
�#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�"
value�"B�" B�"
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
h

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
6
2iter
	3decay
4learning_rate
5momentum
 
8
0
1
2
3
&4
'5
,6
-7
8
0
1
2
3
&4
'5
,6
-7
�
6layer_metrics

7layers
8non_trainable_variables
9layer_regularization_losses
	regularization_losses
:metrics

trainable_variables
	variables
 
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
;layer_metrics

<layers
=non_trainable_variables
>layer_regularization_losses
regularization_losses
?metrics
trainable_variables
	variables
 
 
 
�
@layer_metrics

Alayers
Bnon_trainable_variables
Clayer_regularization_losses
regularization_losses
Dmetrics
trainable_variables
	variables
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Elayer_metrics

Flayers
Gnon_trainable_variables
Hlayer_regularization_losses
regularization_losses
Imetrics
trainable_variables
	variables
 
 
 
�
Jlayer_metrics

Klayers
Lnon_trainable_variables
Mlayer_regularization_losses
regularization_losses
Nmetrics
trainable_variables
 	variables
 
 
 
�
Olayer_metrics

Players
Qnon_trainable_variables
Rlayer_regularization_losses
"regularization_losses
Smetrics
#trainable_variables
$	variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
�
Tlayer_metrics

Ulayers
Vnon_trainable_variables
Wlayer_regularization_losses
(regularization_losses
Xmetrics
)trainable_variables
*	variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

,0
-1

,0
-1
�
Ylayer_metrics

Zlayers
[non_trainable_variables
\layer_regularization_losses
.regularization_losses
]metrics
/trainable_variables
0	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6
 
 

^0
_1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	`total
	acount
b	variables
c	keras_api
D
	dtotal
	ecount
f
_fn_kwargs
g	variables
h	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

b	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

g	variables
�
serving_default_input_4Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_98026
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_98342
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_98400��
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_97725

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_97649

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
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
/__inference_max_pooling2d_7_layer_call_fn_98205

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
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_976712
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_98231

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
K
/__inference_max_pooling2d_6_layer_call_fn_98165

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
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_976492
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_7_layer_call_fn_98189

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_977252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
E
)__inference_flatten_3_layer_call_fn_98225

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
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_977432
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_98220

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_97712

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_98180

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�E
�	
!__inference__traced_restore_98400
file_prefix:
 assignvariableop_conv2d_6_kernel: .
 assignvariableop_1_conv2d_6_bias: <
"assignvariableop_2_conv2d_7_kernel: @.
 assignvariableop_3_conv2d_7_bias:@4
!assignvariableop_4_dense_6_kernel:	� -
assignvariableop_5_dense_6_bias: 3
!assignvariableop_6_dense_7_kernel: 
-
assignvariableop_7_dense_7_bias:
%
assignvariableop_8_sgd_iter:	 &
assignvariableop_9_sgd_decay: /
%assignvariableop_10_sgd_learning_rate: *
 assignvariableop_11_sgd_momentum: #
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: 
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16f
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_17�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
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
�
�
B__inference_dense_6_layer_call_and_return_conditional_losses_97756

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
#__inference_signature_wrapper_98026
input_4!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	� 
	unknown_4: 
	unknown_5: 

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_976402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_4
�

�
,__inference_sequential_3_layer_call_fn_98047

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	� 
	unknown_4: 
	unknown_5: 

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_977802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_conv2d_6_layer_call_fn_98149

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_977022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_97702

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_98215

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_97743

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�:
�
 __inference__wrapped_model_97640
input_4N
4sequential_3_conv2d_6_conv2d_readvariableop_resource: C
5sequential_3_conv2d_6_biasadd_readvariableop_resource: N
4sequential_3_conv2d_7_conv2d_readvariableop_resource: @C
5sequential_3_conv2d_7_biasadd_readvariableop_resource:@F
3sequential_3_dense_6_matmul_readvariableop_resource:	� B
4sequential_3_dense_6_biasadd_readvariableop_resource: E
3sequential_3_dense_7_matmul_readvariableop_resource: 
B
4sequential_3_dense_7_biasadd_readvariableop_resource:

identity��,sequential_3/conv2d_6/BiasAdd/ReadVariableOp�+sequential_3/conv2d_6/Conv2D/ReadVariableOp�,sequential_3/conv2d_7/BiasAdd/ReadVariableOp�+sequential_3/conv2d_7/Conv2D/ReadVariableOp�+sequential_3/dense_6/BiasAdd/ReadVariableOp�*sequential_3/dense_6/MatMul/ReadVariableOp�+sequential_3/dense_7/BiasAdd/ReadVariableOp�*sequential_3/dense_7/MatMul/ReadVariableOp�
+sequential_3/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_3/conv2d_6/Conv2D/ReadVariableOp�
sequential_3/conv2d_6/Conv2DConv2Dinput_43sequential_3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
sequential_3/conv2d_6/Conv2D�
,sequential_3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_3/conv2d_6/BiasAdd/ReadVariableOp�
sequential_3/conv2d_6/BiasAddBiasAdd%sequential_3/conv2d_6/Conv2D:output:04sequential_3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
sequential_3/conv2d_6/BiasAdd�
sequential_3/conv2d_6/ReluRelu&sequential_3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
sequential_3/conv2d_6/Relu�
$sequential_3/max_pooling2d_6/MaxPoolMaxPool(sequential_3/conv2d_6/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_6/MaxPool�
+sequential_3/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+sequential_3/conv2d_7/Conv2D/ReadVariableOp�
sequential_3/conv2d_7/Conv2DConv2D-sequential_3/max_pooling2d_6/MaxPool:output:03sequential_3/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
sequential_3/conv2d_7/Conv2D�
,sequential_3/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/conv2d_7/BiasAdd/ReadVariableOp�
sequential_3/conv2d_7/BiasAddBiasAdd%sequential_3/conv2d_7/Conv2D:output:04sequential_3/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
sequential_3/conv2d_7/BiasAdd�
sequential_3/conv2d_7/ReluRelu&sequential_3/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
sequential_3/conv2d_7/Relu�
$sequential_3/max_pooling2d_7/MaxPoolMaxPool(sequential_3/conv2d_7/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_7/MaxPool�
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
sequential_3/flatten_3/Const�
sequential_3/flatten_3/ReshapeReshape-sequential_3/max_pooling2d_7/MaxPool:output:0%sequential_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2 
sequential_3/flatten_3/Reshape�
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOp�
sequential_3/dense_6/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
sequential_3/dense_6/MatMul�
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_3/dense_6/BiasAdd/ReadVariableOp�
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
sequential_3/dense_6/BiasAdd�
sequential_3/dense_6/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
sequential_3/dense_6/Relu�
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype02,
*sequential_3/dense_7/MatMul/ReadVariableOp�
sequential_3/dense_7/MatMulMatMul'sequential_3/dense_6/Relu:activations:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_3/dense_7/MatMul�
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+sequential_3/dense_7/BiasAdd/ReadVariableOp�
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
sequential_3/dense_7/BiasAdd�
sequential_3/dense_7/SoftmaxSoftmax%sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
sequential_3/dense_7/Softmax�
IdentityIdentity&sequential_3/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp-^sequential_3/conv2d_6/BiasAdd/ReadVariableOp,^sequential_3/conv2d_6/Conv2D/ReadVariableOp-^sequential_3/conv2d_7/BiasAdd/ReadVariableOp,^sequential_3/conv2d_7/Conv2D/ReadVariableOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2\
,sequential_3/conv2d_6/BiasAdd/ReadVariableOp,sequential_3/conv2d_6/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_6/Conv2D/ReadVariableOp+sequential_3/conv2d_6/Conv2D/ReadVariableOp2\
,sequential_3/conv2d_7/BiasAdd/ReadVariableOp,sequential_3/conv2d_7/BiasAdd/ReadVariableOp2Z
+sequential_3/conv2d_7/Conv2D/ReadVariableOp+sequential_3/conv2d_7/Conv2D/ReadVariableOp2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_4
�!
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_97972
input_4(
conv2d_6_97948: 
conv2d_6_97950: (
conv2d_7_97954: @
conv2d_7_97956:@ 
dense_6_97961:	� 
dense_6_97963: 
dense_7_97966: 

dense_7_97968:

identity�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_6_97948conv2d_6_97950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_977022"
 conv2d_6/StatefulPartitionedCall�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_977122!
max_pooling2d_6/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_97954conv2d_7_97956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_977252"
 conv2d_7/StatefulPartitionedCall�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_977352!
max_pooling2d_7/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_977432
flatten_3/PartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_97961dense_6_97963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_977562!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_97966dense_7_97968*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_977732!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_4
�
K
/__inference_max_pooling2d_6_layer_call_fn_98170

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_977122
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:��������� :W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_97671

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
'__inference_dense_7_layer_call_fn_98260

inputs
unknown: 

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_977732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
,__inference_sequential_3_layer_call_fn_97945
input_4!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	� 
	unknown_4: 
	unknown_5: 

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_979052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_4
�
�
'__inference_dense_6_layer_call_fn_98240

inputs
unknown:	� 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_977562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_97905

inputs(
conv2d_6_97881: 
conv2d_6_97883: (
conv2d_7_97887: @
conv2d_7_97889:@ 
dense_6_97894:	� 
dense_6_97896: 
dense_7_97899: 

dense_7_97901:

identity�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_97881conv2d_6_97883*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_977022"
 conv2d_6/StatefulPartitionedCall�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_977122!
max_pooling2d_6/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_97887conv2d_7_97889*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_977252"
 conv2d_7/StatefulPartitionedCall�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_977352!
max_pooling2d_7/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_977432
flatten_3/PartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_97894dense_6_97896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_977562!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_97899dense_7_97901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_977732!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_dense_7_layer_call_and_return_conditional_losses_97773

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�.
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_98140

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @6
(conv2d_7_biasadd_readvariableop_resource:@9
&dense_6_matmul_readvariableop_resource:	� 5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: 
5
'dense_7_biasadd_readvariableop_resource:

identity��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_6/Conv2D�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_6/Relu�
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_7/Conv2D�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_7/Relu�
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_3/Const�
flatten_3/ReshapeReshape max_pooling2d_7/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshape�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_6/MatMul�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_6/Relu�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_7/MatMul�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_7/Softmaxt
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
,__inference_sequential_3_layer_call_fn_97799
input_4!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	� 
	unknown_4: 
	unknown_5: 

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_977802
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_4
�
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_97735

inputs
identity�
MaxPoolMaxPoolinputs*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_dense_6_layer_call_and_return_conditional_losses_98251

inputs1
matmul_readvariableop_resource:	� -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
K
/__inference_max_pooling2d_7_layer_call_fn_98210

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_977352
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_98160

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_97999
input_4(
conv2d_6_97975: 
conv2d_6_97977: (
conv2d_7_97981: @
conv2d_7_97983:@ 
dense_6_97988:	� 
dense_6_97990: 
dense_7_97993: 

dense_7_97995:

identity�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_6_97975conv2d_6_97977*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_977022"
 conv2d_6/StatefulPartitionedCall�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_977122!
max_pooling2d_6/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_97981conv2d_7_97983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_977252"
 conv2d_7/StatefulPartitionedCall�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_977352!
max_pooling2d_7/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_977432
flatten_3/PartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_97988dense_6_97990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_977562!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_97993dense_7_97995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_977732!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_4
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_98200

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
B__inference_dense_7_layer_call_and_return_conditional_losses_98271

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_97780

inputs(
conv2d_6_97703: 
conv2d_6_97705: (
conv2d_7_97726: @
conv2d_7_97728:@ 
dense_6_97757:	� 
dense_6_97759: 
dense_7_97774: 

dense_7_97776:

identity�� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall�dense_6/StatefulPartitionedCall�dense_7/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_97703conv2d_6_97705*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_977022"
 conv2d_6/StatefulPartitionedCall�
max_pooling2d_6/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_977122!
max_pooling2d_6/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_7_97726conv2d_7_97728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_977252"
 conv2d_7/StatefulPartitionedCall�
max_pooling2d_7/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_977352!
max_pooling2d_7/PartitionedCall�
flatten_3/PartitionedCallPartitionedCall(max_pooling2d_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_977432
flatten_3/PartitionedCall�
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_97757dense_6_97759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_977562!
dense_6/StatefulPartitionedCall�
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_97774dense_7_97776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_977732!
dense_7/StatefulPartitionedCall�
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
G__inference_sequential_3_layer_call_and_return_conditional_losses_98104

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @6
(conv2d_7_biasadd_readvariableop_resource:@9
&dense_6_matmul_readvariableop_resource:	� 5
'dense_6_biasadd_readvariableop_resource: 8
&dense_7_matmul_readvariableop_resource: 
5
'dense_7_biasadd_readvariableop_resource:

identity��conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�dense_6/BiasAdd/ReadVariableOp�dense_6/MatMul/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp�dense_7/MatMul/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingSAME*
strides
2
conv2d_6/Conv2D�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_6/BiasAdd/ReadVariableOp�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_6/Relu�
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2DConv2D max_pooling2d_6/MaxPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_7/Conv2D�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
conv2d_7/Relu�
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_3/Const�
flatten_3/ReshapeReshape max_pooling2d_7/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_3/Reshape�
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	� *
dtype02
dense_6/MatMul/ReadVariableOp�
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_6/MatMul�
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_6/BiasAdd/ReadVariableOp�
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_6/Relu�
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype02
dense_7/MatMul/ReadVariableOp�
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_7/MatMul�
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_7/BiasAdd/ReadVariableOp�
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense_7/Softmaxt
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
2

Identity�
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
,__inference_sequential_3_layer_call_fn_98068

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:	� 
	unknown_4: 
	unknown_5: 

	unknown_6:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_979052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_98175

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�)
�
__inference__traced_save_98342
file_prefix.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*x
_input_shapesg
e: : : : @:@:	� : : 
:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	� : 

_output_shapes
: :$ 

_output_shapes

: 
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_48
serving_default_input_4:0���������;
dense_70
StatefulPartitionedCall:0���������
tensorflow/serving/predict:ǅ
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
i_default_save_signature
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_sequential
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
�
regularization_losses
trainable_variables
	variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
�
regularization_losses
trainable_variables
 	variables
!	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
�
"regularization_losses
#trainable_variables
$	variables
%	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
�

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layer
�

,kernel
-bias
.regularization_losses
/trainable_variables
0	variables
1	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
I
2iter
	3decay
4learning_rate
5momentum"
	optimizer
 "
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
�
6layer_metrics

7layers
8non_trainable_variables
9layer_regularization_losses
	regularization_losses
:metrics

trainable_variables
	variables
j__call__
i_default_save_signature
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
,
zserving_default"
signature_map
):' 2conv2d_6/kernel
: 2conv2d_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
;layer_metrics

<layers
=non_trainable_variables
>layer_regularization_losses
regularization_losses
?metrics
trainable_variables
	variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
@layer_metrics

Alayers
Bnon_trainable_variables
Clayer_regularization_losses
regularization_losses
Dmetrics
trainable_variables
	variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_7/kernel
:@2conv2d_7/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Elayer_metrics

Flayers
Gnon_trainable_variables
Hlayer_regularization_losses
regularization_losses
Imetrics
trainable_variables
	variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jlayer_metrics

Klayers
Lnon_trainable_variables
Mlayer_regularization_losses
regularization_losses
Nmetrics
trainable_variables
 	variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Olayer_metrics

Players
Qnon_trainable_variables
Rlayer_regularization_losses
"regularization_losses
Smetrics
#trainable_variables
$	variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
!:	� 2dense_6/kernel
: 2dense_6/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
Tlayer_metrics

Ulayers
Vnon_trainable_variables
Wlayer_regularization_losses
(regularization_losses
Xmetrics
)trainable_variables
*	variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 : 
2dense_7/kernel
:
2dense_7/bias
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
Ylayer_metrics

Zlayers
[non_trainable_variables
\layer_regularization_losses
.regularization_losses
]metrics
/trainable_variables
0	variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
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
N
	`total
	acount
b	variables
c	keras_api"
_tf_keras_metric
^
	dtotal
	ecount
f
_fn_kwargs
g	variables
h	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
`0
a1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
d0
e1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
�B�
 __inference__wrapped_model_97640input_4"�
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
�2�
,__inference_sequential_3_layer_call_fn_97799
,__inference_sequential_3_layer_call_fn_98047
,__inference_sequential_3_layer_call_fn_98068
,__inference_sequential_3_layer_call_fn_97945�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_sequential_3_layer_call_and_return_conditional_losses_98104
G__inference_sequential_3_layer_call_and_return_conditional_losses_98140
G__inference_sequential_3_layer_call_and_return_conditional_losses_97972
G__inference_sequential_3_layer_call_and_return_conditional_losses_97999�
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

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_conv2d_6_layer_call_fn_98149�
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
�2�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_98160�
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
�2�
/__inference_max_pooling2d_6_layer_call_fn_98165
/__inference_max_pooling2d_6_layer_call_fn_98170�
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
�2�
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_98175
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_98180�
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
�2�
(__inference_conv2d_7_layer_call_fn_98189�
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
�2�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_98200�
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
�2�
/__inference_max_pooling2d_7_layer_call_fn_98205
/__inference_max_pooling2d_7_layer_call_fn_98210�
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
�2�
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_98215
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_98220�
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
�2�
)__inference_flatten_3_layer_call_fn_98225�
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
�2�
D__inference_flatten_3_layer_call_and_return_conditional_losses_98231�
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
�2�
'__inference_dense_6_layer_call_fn_98240�
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
�2�
B__inference_dense_6_layer_call_and_return_conditional_losses_98251�
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
�2�
'__inference_dense_7_layer_call_fn_98260�
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
�2�
B__inference_dense_7_layer_call_and_return_conditional_losses_98271�
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
#__inference_signature_wrapper_98026input_4"�
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
 �
 __inference__wrapped_model_97640w&',-8�5
.�+
)�&
input_4���������
� "1�.
,
dense_7!�
dense_7���������
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_98160l7�4
-�*
(�%
inputs���������
� "-�*
#� 
0��������� 
� �
(__inference_conv2d_6_layer_call_fn_98149_7�4
-�*
(�%
inputs���������
� " ���������� �
C__inference_conv2d_7_layer_call_and_return_conditional_losses_98200l7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0���������@
� �
(__inference_conv2d_7_layer_call_fn_98189_7�4
-�*
(�%
inputs��������� 
� " ����������@�
B__inference_dense_6_layer_call_and_return_conditional_losses_98251]&'0�-
&�#
!�
inputs����������
� "%�"
�
0��������� 
� {
'__inference_dense_6_layer_call_fn_98240P&'0�-
&�#
!�
inputs����������
� "���������� �
B__inference_dense_7_layer_call_and_return_conditional_losses_98271\,-/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������

� z
'__inference_dense_7_layer_call_fn_98260O,-/�,
%�"
 �
inputs��������� 
� "����������
�
D__inference_flatten_3_layer_call_and_return_conditional_losses_98231a7�4
-�*
(�%
inputs���������@
� "&�#
�
0����������
� �
)__inference_flatten_3_layer_call_fn_98225T7�4
-�*
(�%
inputs���������@
� "������������
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_98175�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_98180h7�4
-�*
(�%
inputs��������� 
� "-�*
#� 
0��������� 
� �
/__inference_max_pooling2d_6_layer_call_fn_98165�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
/__inference_max_pooling2d_6_layer_call_fn_98170[7�4
-�*
(�%
inputs��������� 
� " ���������� �
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_98215�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_98220h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
/__inference_max_pooling2d_7_layer_call_fn_98205�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
/__inference_max_pooling2d_7_layer_call_fn_98210[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_sequential_3_layer_call_and_return_conditional_losses_97972s&',-@�=
6�3
)�&
input_4���������
p 

 
� "%�"
�
0���������

� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_97999s&',-@�=
6�3
)�&
input_4���������
p

 
� "%�"
�
0���������

� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_98104r&',-?�<
5�2
(�%
inputs���������
p 

 
� "%�"
�
0���������

� �
G__inference_sequential_3_layer_call_and_return_conditional_losses_98140r&',-?�<
5�2
(�%
inputs���������
p

 
� "%�"
�
0���������

� �
,__inference_sequential_3_layer_call_fn_97799f&',-@�=
6�3
)�&
input_4���������
p 

 
� "����������
�
,__inference_sequential_3_layer_call_fn_97945f&',-@�=
6�3
)�&
input_4���������
p

 
� "����������
�
,__inference_sequential_3_layer_call_fn_98047e&',-?�<
5�2
(�%
inputs���������
p 

 
� "����������
�
,__inference_sequential_3_layer_call_fn_98068e&',-?�<
5�2
(�%
inputs���������
p

 
� "����������
�
#__inference_signature_wrapper_98026�&',-C�@
� 
9�6
4
input_4)�&
input_4���������"1�.
,
dense_7!�
dense_7���������
