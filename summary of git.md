# Summary of git

* 删除git历史大文件记录导致的项目臃肿问题
    - 1.查看历史提交大文件的记录（前10大文件）
        - `git rev-list --objects --all | grep "$(git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -n | tail -10 | awk '{print$1}')"`
    - 2.删除你想删除的文件记录
        - `git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch opcode_cnn/train_data.csv' --prune-empty --tag-name-filter cat -- --all`
    - 3.强制提交修改
        - `git push origin master --force`

    - 如果遇到拒绝提交则可能需要把项目的protect属性改下。

* git多账号配置等
    - https://www.cnblogs.com/elisun/p/6881612.html

    - http方式记住密码
        - https://www.jianshu.com/p/c41f18f62858

    - https://www.jianshu.com/p/7d57ce4147d3

