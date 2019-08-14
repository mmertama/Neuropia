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
        this.ctx.lineWidth = 16;
        this.ctx.stroke();
        this.ctx.closePath();

    }
    
    erase() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    //    document.getElementById(this.element).style.display = "none";
     //   document.getElementById(this.element).style.display = "display";
    } 
    

    
    getData(width, height) {
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
    }
}

