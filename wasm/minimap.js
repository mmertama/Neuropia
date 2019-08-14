function _pad(h){ 
    return (h.length==1) ? "0"+h : h;
}

class Minimap {
    constructor(element, data) {
        this.element = document.getElementById(element);
        this.ctx = this.element.getContext("2d");
        this.setData(data);
        }
    draw() {
        if(this.data !== undefined && this.data.length > 0) {
            const dim = Math.sqrt(this.data.length);
            const width = this.element.width / dim;
            const height = this.element.height / dim;
            let p = 0;
            let posy = 0;
        
            
            for(let j = 0; j < dim; j++) {
                let posx = 0;
                for(let i = 0; i < dim; i++) {
                    const v = this.data[p];
                    const hex = Number(v).toString(16);
                    
                    const color = "#" + _pad(hex) + _pad(hex) + _pad(hex);
                    this.ctx.fillStyle = color;
                    this.ctx.fillRect(posx, posy, width, height);
                    p++;
                    posx += width;
                }
                posy += height;
            }
        }
    }
    erase() {
        this.ctx.clearRect(0, 0, this.element.width, this.element.height);
    }
    setData(data) {
        this.ctx.clearRect(0, 0, this.element.width, this.element.height);
        this.data = data;
        this.draw();
    }
}
    