# G19-INPUT：因果前端与失败账本

本轮B负责前端输入与审计，A/B深度/位姿/语义格式不改；不启动新Gaussian训练、不修改G18-D。首先复核旧G14的frame_features.csv：1000行tracking_state=2（源码枚举OK），作为旧输入的描述性失败账本保留。旧最终轨迹仍不能替代即时位姿。

固定KITTI08源3000–3999，复用G14的1000对image_2/image_3、时间戳、标定、settings.yaml、O0策略、ORB核心库和词典。所有2000图像SHA、原参数和私有库锁定；标定数值与原G14一致，本轮只补输入记录，不重新校准或改前端参数。图像逐帧解码，每次TrackStereo返回后、下一帧读入前封存其Tcw、以FP64三维逆转换的C2w、tracking_state、map_id和MapChanged标志。

不改ORB库，用独立C++观察程序，viewer关闭。保留原所有帧与状态，不裁掉初始化/丢失帧。仅当1000帧都为OK、位姿有限且近SE3、单一已知map_id时，输出可供映射的12列external_causal_P2_c2w.txt；否则完整账本保留，主位姿文件不发布。RECENTLY_LOST/LOST/未初始化/异常状态、地图变化分别统计，不把refiner回退等同tracking-lost。

strace跟踪所有线程的openat/write，验证每帧账本写入前只打开不晚于该帧的左右目、两目均已读入、下一帧图像在前帧账本封存后才读取；禁止GT路径。全部1000返回Tcw与C2w独立CPU复算，尺度固定1；停机后轨迹另存offline_final_NOT_FOR_MAPPING，只作返回后优化改写量的无GT诊断，禁止用它替换因果输入。

预算：构建600秒、完整1000帧1800秒、任务新文件1GiB、数据盘保留6GiB。CPU与D并行，时间只用于预算，不作FPS比较。若失败保留原源码/构建/运行/审计日志，不因观察超时重跑。产物是前端输入合同，不能单独证明严格在线3DGS系统、past-only置信度或正式两窗三seed质量通过。
