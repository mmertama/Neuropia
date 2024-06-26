   var painter = null;
            var histogram = null;
            var neuropia = null;
            var minimap = null;
            var Module = {
        onRuntimeInitialized: function() {
            histogram = new Histogram('histogram', 3);
            histogram.setColors([], "Green");
            histogram.setLabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            minimap = new Minimap('minimap'); 
            painter = new Painter('canvas', function (p) {
                const d = p.getData(28, 28);
                if(d == undefined)
                    return;
                minimap.setData(d);
                if(Module.isNetworkValid(neuropia)) { 
                    const result = Module.feed(neuropia, d);
                    let max = 0;
                    let maxi =0;
                    let data = [];
                    for(let k = 0; k < result.size(); k++) {
                        const r = result.get(k);
                        if(r > max) {
                            maxi = k;
                            max = r;
                        }
                        data.push(r)
                      //  console.log(k + ": " + r);
                    } 
                    histogram.setData(data);
                    document.getElementById('guess').innerHTML = maxi + ' with likehood ' + max.toFixed(3);                    
                }
            }); 
            
            neuropia = Module.create(".");
            const terminal = document.getElementById("terminal")
            Module.setLogger(neuropia, setLog); 
            setLogl("Neuropia constructed");
            const fopen = Module.load(neuropia, "default_out.bin");
            document.getElementById("verifyBtn").disabled = !fopen;
            if(fopen)
                setLogl("Demo network loaded" + String.fromCharCode(10));
           
           
            Module.setParam(neuropia, "Images", "mnist/train-images-idx3-ubyte");
            Module.setParam(neuropia, "ImagesVerify", "mnist/t10k-images-idx3-ubyte");
            Module.setParam(neuropia, "Labels", "mnist/train-labels-idx1-ubyte");
            Module.setParam(neuropia, "LabelsVerify", "mnist/t10k-labels-idx1-ubyte");
            Module.setParam(neuropia, "Classes", "10");
            
            const params = Module.params(neuropia); 
            const keys = params.keys();
            const paramList = document.getElementById("paramList");
            for (let i = 0; i < keys.size(); i++) {
                const k = keys.get(i);
                const p = params.get(k);
                const type = p.get(0);
                const value = p.get(1);
                const pattern = p.get(2);
               
                const input = document.createElement("INPUT");
                input.setAttribute("name", k)
                input.setAttribute("class", "param");
                switch (type) {
                    case 'interger number':
                        input.setAttribute("type", "number");
                        break;
                     case 'real number':
                        input.setAttribute("type", "number");
                        input.setAttribute("step", "any");
                        break;
                    case 'string':
                        input.setAttribute("type", "text");
                        input.setAttribute("mandatory", "false");
                        break;
                    case 'file':
                      //  input.setAttribute("type", "file");
                      //  break;
                        continue;
                    case 'true|false':
                        input.setAttribute("type", "checkbox");
                        break;
                    default:
                        input.setAttribute("type", "text");
                        input.setAttribute("pattern", pattern);
                }
                input.setAttribute("value", value);
                const th1 = document.createElement("th");
                th1.textContent = k;
                const th2 = document.createElement("th");
                th2.appendChild(input);
                
                const tr = document.createElement("tr");
                const th3 = document.createElement("th");
                th3.setAttribute("name", "name_" + k);
                th3.textContent = type;
                tr.appendChild(th1);
                tr.appendChild(th2);
                tr.appendChild(th3);
                paramList.appendChild(tr)
            }
            
            if(!fopen) {
                setLogl("Demo network is missing, train one, please wait");
                startTrain();
            }

         //   if(!Module.showImage(neuropia, "train-images-idx3-ubyte", 0))
        //        console.log("OHO")
        }
    };
        
        /* Functions */

        function uiNumber(value) {
            var val = value + "";
            var pos = Utils.find(val, '.') + 3; // 3 == two digits
            return Utils.substr(val, 0, pos);
        }

         function setLog(str) {
                const  r = str.indexOf('\r');
                if(r < 0) {
                    terminal.value += str;
                } else {
                    const lastNewLine = terminal.value.lastIndexOf('\n');
                    terminal.value = terminal.value.slice(0, lastNewLine) + str;
                }
            }
        
        
        function setLogl(str) {
            setLog(str + '\n')
        }
        
        function logMEMFS(directory) {
            let entries = [];
            try {
                entries = FS.readdir(directory);
            } catch(err) { // just ignore
                return;
            }
            entries.forEach(function(entry) {
                if(entry[0] == '.')
                    return;
                const path = directory + '/' + entry;
                console.log(path);
                if (FS.isDir(FS.lookupPath(path).node.mode)) {
                    logMEMFS(path);
                }
            });
        }
                          
        function startTrain() {
            document.getElementById("verifyBtn").disabled = true;
            const params = document.getElementsByClassName("param");
            let ok = true;
            const plist = {}
            for (let i = 0, length = params.length; i < length; i++) {
                const name = params[i].name;
                const value = params[i].value;
                plist[name] = value;
                const isOk = Module.setParam(neuropia, name, value);
                ok = ok && isOk;
                let status = document.getElementsByName("name_" + name); 
                if(!isOk) {
                    status[0].setAttribute("style", "color:#ff0000");
                    setLogl("Bad parameter " + name);
                }
            } 
            if(ok) {
                const iterations = Module.params(neuropia).get('Iterations').get(1);
                const batch_size = Module.batchSize();
                console.log("Ready for Training", iterations, batch_size);
                console.assert(batch_size > 0);
                if(iterations === 0)
                    return;
                let iteration = 0;
                function doTrain() {
                    if(iteration < iterations && Module.train(neuropia, batch_size)) {
                        setTimeout(doTrain, 0);
                        iteration += batch_size;
                    } else {
                        if(Module.isNetworkValid(neuropia)) {
                            console.log("Training done");
                            document.getElementById("verifyBtn").disabled = false;
                        }
                        else {
                            setLogl("Training failed");
                            logMEMFS("/");
                        }
                    }
                }
                doTrain();
            }
        }
            
        function verify() {
            let iteration = 0;
            console.log("Start verify");
            function doVerify() {
                if(Module.verify(neuropia, iteration)) {
                    setTimeout(doVerify, 0);
                    ++iteration;
                } else {
                    const verifyResult = Module.verifyResult(neuropia);
                    if(verifyResult >= 0) {
                        console.log("Verify done");
                        document.getElementById("verificationResult").innerHTML = "Verification: " +  uiNumber(verifyResult * 100) + "%";
                    } else {
                        alert("Training failed");
                    }
                }
            }
            doVerify();
            return false;
        }
    