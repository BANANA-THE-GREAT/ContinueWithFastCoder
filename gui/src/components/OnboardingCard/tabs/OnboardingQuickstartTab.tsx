import ContinueLogo from "../../gui/ContinueLogo";
import QuickStartSubmitButton from "../components/QuickStartSubmitButton";

function OnboardingQuickstartTab() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="xs:px-0 flex w-full max-w-full flex-col items-center justify-center px-4 text-center">
        <div className="xs:flex hidden">
          <ContinueLogo />
          {/* <ContinueLogo height={75} /> */}
        </div>

        {/* <p className="xs:w-3/4 w-full text-sm">
          Quickly get up and running using our API keys. After this trial, we'll
          help you set up your own models.
        </p>

        <p className="xs:w-3/4 w-full text-sm">
          To prevent abuse, we'll ask you to sign in to GitHub.
        </p> */}

        {/* <QuickStartSubmitButton /> */}

        <p className="w-full text-sm mb-0">
          {/* <b>您可以在上方输入框内输入指令：</b> */}
          <b>Please enter instructions in the input box above:</b>
        </p>

        <p className="w-full text-sm mb-0">
          <code>/renew</code>
          <br/>
          {/* 检索当前工作区中的所有 python 文件，来重新构建仓库代码检索库 */}
          Retrieve all python files in the current workspace to rebuild the repository code datastore
        </p>

        <p className="w-full text-sm mb-0">
          <code>/enableAcc</code>
          <br/>
          {/* 开启 CodeSwift 加速方法 */}
          Enable FastCoder acceleration method
        </p>

        <p className="w-full text-sm mb-0">
          <code>/disableAcc</code>
          <br/>
          {/* 关闭 CodeSwift 加速方法 */}
          Disable FastCoder acceleration method
        </p>

        <br/>
        <p className="w-full text-sm mb-0">
          {/* <b>或者选中一段代码，右键单击之后选择 Continue 下的选项：</b> */}
          <b>Or select a piece of code, right-click and choose the option under `Continue`:</b>
        </p>

        <p className="w-full text-sm mb-0">
          <code>Enable displaying code source</code>
          <br/>
          {/* 显示选中部分补全过程中的来源 */}
          Display the source of the selected code during the completion process
          <br/>
          {/* 【快捷键：<code>ctrl + E</code>】 */}

          <span style={{ color: 'rgba(0, 83, 200, 0.9)' }}>Blue indicates code from cache</span>
          <br/>
          <span style={{ color: 'rgba(200, 30, 30, 0.9)' }}>Red indicates code from the retrieval library</span>
          <br/>
          <span style={{ color: 'rgba(0, 120, 100, 0.9)' }}>Green indicates code generated directly by the model</span>
          <br/>
          {/* <span style={{ color: 'rgba(0, 83, 200, 0.9)' }}>蓝色表示来自缓存</span>
          <br/>
          <span style={{ color: 'rgba(200, 30, 30, 0.9)' }}>红色表示来自检索库</span>
          <br/>
          <span style={{ color: 'rgba(0, 120, 100, 0.9)' }}>绿色表示由模型直接生成</span> */}
        </p>
        <br/>

        <p className="w-full text-sm mb-0">
          <code>Disable displaying code source</code>
          <br/>
          {/* 取消选中部分的来源显示 */}
          Disable the display of the source of the selected code
          <br/>
          {/* 【快捷键：<code>ctrl + N</code>】 */}
        </p>
        
        <br/>
        <hr className="w-full  mt-4" />
      </div>
    </div>
  );
}

export default OnboardingQuickstartTab;
