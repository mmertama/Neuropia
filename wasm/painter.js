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

function _printimage(c)  {
        let p = 0;
        for(let j = 0; j < 28; j++) {
            let s = j + ":"
            for(let i = 0; i < 28; i++) {
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
    

    _getImageCentered() {
        const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
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
        
        const width = maxX  - minX;
        const height = maxY - minY;        
        const startX = (this.canvas.width - width) / 2;
        const startY = (this.canvas.width - height) / 2;
                        
       
        const buffer = new ArrayBuffer(this.canvas.width * this.canvas.height);
        const copy = new Uint8Array(buffer);
        
        for(let j = minY; j < maxY; j++) {
            let p = j * stride + minX * 4;
            let pp = startY * stride + startX * 4;
            for(let i = minX; i < maxX; i++) {
                copy[pp] = imgData.data[p];
                p += 4;
                pp++;
            }
        }
        
        return copy;
    }
    
    getData(width, height) {
        
        const imgData = this._getImageCentered();
            
        const wsamp = this.canvas.width / width;
        const hsamp = this.canvas.height / height;
        
        const data = new ArrayBuffer(width * height);
        
        const stride = this.canvas.width;
       
        for(let j = 0 ; j < height; j++) {
            for(let i = 0 ; i < width; i++) {
                let sampled = 0;
                const gp = i + j * width;
                const off =  (i * wsamp) + (j * stride * hsamp)
                for(let y = 0; y < hsamp; y++) {
                    const p = (y * stride) + off;
                    for(let x = 0; x < wsamp; x++) {
                        const pos = x + p;  
                        const a = imgData[pos];
                        sampled += a;
                    }
                }
                data[gp] = Math.round(sampled / (hsamp * wsamp));
            }
        }
     //   _printimage(data);
        return data;
    }
    
 /*   
    getData(width, height) {
        
        const ext = this._getExtents()
        
        const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        
        const wsamp = this.canvas.width / width;
        const hsamp = this.canvas.height / height;
        
        const data = new Array(width * height);
        
        const stride = imgData.width * 4;
       
        for(let j = 0 ; j < height; j++) {
            for(let i = 0 ; i < width; i++) {
                let sampled = 0;
                const gp = i + j * width;
                const off =  (4 * i * wsamp) + (j * stride * hsamp)
                for(let y = 0; y < hsamp; y++) {
                    const p = (y * stride) + off;
                    for(let x = 0; x < wsamp * 4; x += 4) {
                        const pos = x + p;  
                        const r = imgData.data[pos + 0];
                        const g = imgData.data[pos + 1];
                        const b = imgData.data[pos + 2];
                        const a = imgData.data[pos + 3];
                        
                        sampled += a;// ((a * ((r * 0.3 + g * 0.59 + b * 0.11)))) / 255 ;
                    }
                }
                data[gp] = Math.round(sampled / (hsamp * wsamp));
            }
        }
     //   _printimage(data);
        return data;
    }*/
}

