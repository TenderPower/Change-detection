   function Person(name,age){
      debugger
      this.name = name
      this.age = age
    }
    // 动态方法:需要实例化
    Person.prototype.show = function(){
      debugger
      console.log(this.name,this.age)
    }
    console.log(Person)
    debugger
    let per = new Person('gsy','18')
   console.log(per)
    console.log(per.__proto__)
    // 静态方法：不需要实例化
    Person.eat = function(){
      console.log('eat--',this.name,this.age,this)
    }
    console.log(Person)
    debugger
    Person.eat();
