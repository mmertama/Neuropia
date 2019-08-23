function _getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left,
      y: evt.clientY - rect.top
    };
}

function _mapGrey(m) {
        const greys = "`.:-=+*#%@";
        const p = m / 26;
        return greys.charAt(9 - p);
    }

function _printimage(c, w, h)  {
        let p = 0;
        for(let j = 0; j < h; j++) {
            let s = j + ":"
            for(let i = 0; i < w; i++) {
                s += _mapGrey(c[p]);
                ++p;
            }
        console.log(s);
        }
    }

class Painter {
    constructor(element, onStop) {
        const self = this;
        this.prevX = 0;
        this.currX = 0;
        this.prevY = 0;
        this.currY = 0;
        this.isDraw = false;
        this.element = element;
        this.onStop = onStop;
        this.canvas = document.getElementById(element);
        this.ctx = canvas.getContext("2d");
        this.canvas.addEventListener("mousemove", function (e) {
            if(self.isDraw) {
                self.prevX = self.currX;
                self.prevY = self.currY;
                const p = _getMousePos(this, e);
                self.currX = p.x;
                self.currY = p.y;
                self.draw();
            }
        }, false);
        this.canvas.addEventListener("mousedown", function (e) {
            const p = _getMousePos(this, e);
            if(p.x >= 0 && p.y >= 0 && p.x < self.canvas.width && p.y < self.canvas.height && !self.isDraw) {
                self.isDraw = true;
                const p = _getMousePos(this, e);
                self.currX = p.x;
                self.currY = p.y;
            }
        }, false);
        this.canvas.addEventListener("mouseup", function (e) {
            self.isDraw = false;
        }, false);
        this.canvas.addEventListener("mouseout", function (e) {
            self.isDraw = false;
            self.onStop(self);
        }, false);
    }
    
    draw() {
        this.ctx.beginPath();
        this.ctx.moveTo(this.prevX, this.prevY);
        this.ctx.lineTo(this.currX, this.currY);
        this.ctx.strokeStyle = "Black";
        this.ctx.lineCap = "round";
        this.ctx.lineJoin = "round"
        this.ctx.lineWidth = 20;
        this.ctx.stroke();
        this.ctx.closePath();

    }
    
    erase() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    //    document.getElementById(this.element).style.display = "none";
     //   document.getElementById(this.element).style.display = "display";
    } 
    
    _findExtens(imgData) {
        const stride = imgData.width * 4;
        let minX = this.canvas.width;
        let maxX = 0
        let minY = this.canvas.height;
        let maxY = 0
        
        let p = 3;
        for(let j = 0; j <  this.canvas.height; j++) {
            for(let i = 0; i < this.canvas.width; i++) {
                const a = imgData.data[p];
                if(a > 0) {
                    if(i < minX) minX = i;
                    if(i > maxX) maxX = i;
                    if(j < minY) minY = j;
                    if(j > maxY) maxY = j;
                }
                p += 4;
            }
        }
        return {'x': minX, 'y': minY, 'width': maxX - minX, 'height': maxY - minY};
    }
    
    _crop(data, dataWidth, dataHeight, width, height) {
        const buffer = new ArrayBuffer(width * height);
        const copy = new Uint8Array(buffer);
        const startX = Math.floor((dataWidth - width) / 2);
        const startY = Math.floor((dataHeight - height) / 2);
        let p = 0;
        let pp = startY * dataWidth + startX;
        for(let j = 0; j < height; j++) {
            for(let i = 0; i < width; i++) {
                copy[p] = data[pp];
                p++;
                pp++;
            }
            pp += dataWidth - width;
        }
        return copy;
    }
    
    _getImageCentered() {
        const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

        const ext = this._findExtens(imgData);
            
        const startX = Math.floor((this.canvas.width - ext.width) / 2);
        const startY = Math.floor((this.canvas.width - ext.height) / 2);
                        
       
        const buffer = new ArrayBuffer(this.canvas.width * this.canvas.height);
        const copy = new Uint8Array(buffer);
        
        const stride = imgData.width * 4;
        let pp = startY * this.canvas.width + startX;
        let p = (ext.y * stride) + (ext.x * 4) + 3; 
        for(let j = 0; j < ext.height; j++) {
            for(let i = 0; i < ext.width; i++) {
                copy[pp] = imgData.data[p];
                p += 4;
                pp++;
            }
            pp += this.canvas.width - ext.width;
            p += stride - ext.width * 4;
        }
        
        //rescale if needed
        //since the image is now centered we can just crop it and then do actual scaling on 
        //resampling
        const scaleFactor = Math.max(ext.width / this.canvas.width, ext.height / this.canvas.height);
        
        if(scaleFactor < 0.1)
            return undefined;
       
        if(scaleFactor > 0.7) {
            return {'data': copy, 'width': this.canvas.width, 'height': this.canvas.height};
        } 
        
        const width = Math.ceil(this.canvas.width * (scaleFactor * 1.5));
        const height = Math.ceil(this.canvas.height * (scaleFactor * 1.5));
        return {'data': this._crop(copy, this.canvas.width, this.canvas.height, width, height), 'width': width, 'height': height};
    }
    
    getData(width, height) {
        
        const imgData = this._getImageCentered();
        
        if(imgData == undefined || !width || !height)
            return undefined;
            
        const wsamp = Math.max(1, Math.floor(imgData.width / width));
        const hsamp = Math.max(1, Math.floor(imgData.height / height));
        
        const data = new Array(width * height);
        
        const stride = imgData.width;
       
        for(let j = 0 ; j < height; j++) {
            for(let i = 0 ; i < width; i++) {
                let sampled = 0;
                const gp = i + j * width;
                const off =  (i * wsamp) + (j * stride * hsamp)
                for(let y = 0; y < hsamp; y++) {
                    const p = Math.floor((y * stride) + off);
                    for(let x = 0; x < wsamp; x++) {
                        const pos = x + p;  
                        const a = imgData.data[pos];
                        sampled += a;
                    }
                }
                data[gp] = Math.round(sampled / (hsamp * wsamp));
            }
        }
        _printimage(data, width, height);
        return data;
    }
}

