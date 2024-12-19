## 1.下载并安装ollama(Docker镜像中实现)
curl -fsSL https://ollama.com/install.sh | sh
## 2.设置环境变量
### 设置默认端口
echo export OLLAMA_HOST="0.0.0.0:11434">>~/.bashrc
### 设置模型默认下载地址
echo export OLLAMA_MODELS=/root/ollama/models>>~/.bashrc
### 查看设置情况
cat ~/.bashrc
### 激活配置
source ~/.bashrc
### 启动Ollama服务
ollama serve
## 3.模型创建与运行
### (1).开源模型
重新开启一个shell终端，执行ollama run 模型名称(HuggingFace上可以搜到的开源模型基本上都可以直接run，会对开源模型先下载再执行，所以第一次执行时间会比较长)
### (2).微调模型
这里需要将微调后的模型先转化为ollama支持的形式(这里以gguf格式为例进行说明)

####  a.安装cmake<br>
1. wget https://cmake.org/files/v3.21/cmake-3.21.1.tar.gz
2. 修改CMakeLists.txt，在其中添加set(CMAKE_USE_OPENSSL OFF)，可以直接添加到文件首行， 这里具体位置应该没有要求。用于解决没有openssl的问题。
3. 使用如下命令重载make的指定路径,如果没有sudo权限，直接写入usr会报没有权限的异常。其中$HOME是指向你该用户的root路径，make install之后的bin文件等就保存在你指定的anyDirectory目录下。用于解决没有sudo权限问题。
<br>    ./configure --prefix=$HOME/anyDirectory
   <br> make
    <br>make install
4. 在完成上述操作之后，就需要配置本用户的.bashrc文件，在其中添加 export PATH="$HOME/anyDirectory/bin:$PATH" 并重新加载.bashrc文件。这里的anyDirectory需要跟上面--prefix时候设置的保持一致。

#### b.基于llama.cpp转换模型的格式<br>
1. git clone https://github.com/ggerganov/llama.cpp
<br>cd llama.cpp
<br>pip install -r requirements/requirements_convert_hf_to_gguf.txt
<br>cmake -B build
<br>cmake --build build --config Release

2. 在llama.cpp目录下执行转换
<br>python convert_hf_to_gguf.py /home/intern/LYZ/大模型微调/llama-3-chinese-8b-instruct_chinese_merged --outtype f16 --outfile /home/intern/LYZ/大模型微调/llama__chinese.gguf

3. 量化(非必要步骤)
<br>cd build/bin/Release
<br>./llama-quantize /home/intern/LYZ/大模型微调/llama-3-chinese-8b-instruct-LoRA-merged.gguf /home/intern/LYZ/大模型微调/llama-3-chinese-8b-instruct-LRA-merged-quantiez.gguf q4_0

#### c.模型部署
1. 首先将格式为gguf的模型与Modelfile文件放入同一文件夹下(不同模型需要修改Modelfile文件第一行的文件路径)
2. cd到该文件目录下
2. ollama create 自定义模型名称 -f Modelfile
3. ollama run 自定义模型名称

## 4. 模型删除
ollama rm 模型名称
## 5. Open-WebUI的部署
如果Ollama在另一台服务器上，请使用以下命令：<br>
docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=https://example.com -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main(连接到另一台服务器上的Ollama时，请将OLLAMA_BASE_URL更改为服务器的URL(即服务器IP))<br>
安装完成后，您可以通过http://服务器IP:3000访问OpenWebUI(以3090集群为例IP即为10.9.113.99)


