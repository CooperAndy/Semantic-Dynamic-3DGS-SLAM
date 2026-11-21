# 车辆区域状态复用工具

先读项目中的 `特调地图/车辆状态成功经验与跨序列复用_20261114.md`。

- `run_vehicle_objects.py --profile PROFILE --output NEW_DIR`：只计算，不自动修改 Demo。
- `seq09_profile.json`：已经精确回归成功 seq09 的配置。
- `seq04_TEMPLATE.json`、`seq17_TEMPLATE.json`：缺少该序列自己的实例输入与观测目录，填写并核验后才运行。
- `prepare_inputs.py --help`：按明确的原始/本地帧号和实际分辨率导出 mask 内原始深度及实例数据。
- `core/`：冻结成功版本的算法函数。通用入口导入这些函数，不调用参考脚本中 seq09 专用的 main。
- `seq09_regression_check.json`：通用入口与原成功输出逐项相同。
- `prepare_validation_check.json`：准备器的分辨率、编号偏移和深度值保留验证；不是其他真实序列的效果评价。

当前只支持与 seq09 相同的 Cityscapes-19 类别约定、米制 Y 竖直地图坐标、同索引语义图/几何和原实例输出格式。输入不匹配时会停止；其他地图不能仅改序列名后安装。相机、mask、深度与图像尺寸从输入读取，未写死 seq09 的 1226×370。

当场景存在真实起步、停车或远处动态漏检时，需重新验证或关闭 `enable_static_hold`。不要把所有车辆都判静态当成成功目标。

冻结成功包位于上级 `seq09_success_20261114/`；当前已接受的 Demo 不会被本工具改动。
