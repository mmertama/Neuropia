class Histogram {
    constructor(element, gap, data) {
        this.gap = gap || 3;
        this.element = document.getElementById(element);
        this.ctx = this.element.getContext("2d");
        this.textPos = 10;
        this.ctx.font = "12px Georgia";
        this.setData(data);
        }
    
    draw() {
     if(this.data !== undefined && this.data.length > 0) {
        const width = (this.element.width - (this.gap * this.data.length - 1)) / this.data.length; 
        let posx = 0;
        for(let i = 0; i < this.data.length; i++) {
            this.ctx.strokeStyle = "Black";
            this.ctx.fillStyle = "Black";
            if(this.colors !== undefined && i < this.colors.length)
                this.ctx.fillStyle = this.colors[i]
            if(this.maxColor !== undefined && i == this.max)
                this.ctx.fillStyle = this.maxColor;
            if(this.minColor !== undefined && i == this.min)
                this.ctx.fillStyle = this.minColor;
            const height = this.element.height * this.data[i];
            this.ctx.fillRect(posx, this.element.height - height, width, height);
            if(this.labels !== undefined && i < this.labels.length) {
                this.ctx.fillStyle = "Black";
                this.ctx.strokeStyle = "White";
                this.ctx.textAlign = "center"; 
                this.ctx.fillText(this.labels[i], posx + width / 2, this.textPos); 
                }
            posx += width + this.gap;
            }
        }
    }
        
    setData(data) {
        if(data !== undefined && data.length > 0) {
            let max = 0;
            this.maxi = 0;
            let min = 0xFFFFFFF;
            this.mini = 0;
            for(let i = 0; i < data.length; i++) {
                if(data[i] > max) {
                    max = data[i];
                    this.max = i;
                }
                if(data[i] < min) {
                    min = data[i];
                    this.min = i;
                }
            }
            
        }
        this.data = data;
        this.erase();
        this.draw();
    }
    
     erase() {
        this.ctx.clearRect(0, 0, this.element.width, this.element.height);
    //    document.getElementById(this.element).style.display = "none";
     //   document.getElementById(this.element).style.display = "display";
    } 
        
    setColors(colors, max, min) {
        this.colors = colors;
        this.maxColor = max;
        this.minColor = min;
        this.draw();
    }
        
    setLabels(labels, textPos) {
        this.labels = labels;
        this.textPos = textPos || this.textPos;
        this.erase();
        this.draw();
    }    
}